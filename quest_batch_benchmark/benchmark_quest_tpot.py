#!/usr/bin/env python3
"""Quest decode-speed (TPOT) benchmark at fixed decode batch sizes.

Boots ONE sglang ModelRunner (via sglang.bench_one_batch) for a single
attention mode (`quest` or `dense`). Then, for each batch size in
ascending order, it:

  * replicates the benchmark request `batch_size` times into Req objects,
  * prefills the batch with extend(),
  * runs `warmup` then `measured` decode steps,
  * times every measured decode step with CUDA events.

A decode step emits exactly one output token per request, so the step
latency *is* the time-per-output-token (TPOT). One CSV row is written per
measured step. On CUDA OOM the batch size is recorded with status=oom and
the (ascending) sweep stops — larger batches would also OOM.

KV-cache reuse is disabled (disable_radix_cache=True) so quest and dense
are compared on equal footing.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch

import vortex_torch  # noqa: F401  -- registers the vortex attention backend
from vortex_torch.engine.sgl import DEFAULT_SCHEDULE_POLICY

from sglang.bench_one_batch import decode, extend, load_model
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger

from transformers import AutoTokenizer

from prompt_io import build_input_ids

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
QUEST_MODULE = "gqa_quest_sparse_attention"

RAW_CSV_FIELDS = [
    "run_timestamp", "attention", "batch_size", "model", "topk_val",
    "input_tokens", "warmup_steps", "measured_steps", "step_idx",
    "step_latency_ms", "status",
]


def summarize_step_latencies(step_ms: List[float]) -> dict:
    """Aggregate per-step decode latencies (ms) into TPOT statistics."""
    ordered = sorted(step_ms)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        idx = (n - 1) * p
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    return {
        "tpot_ms_mean": statistics.fmean(step_ms),
        "tpot_ms_p50": pct(0.50),
        "tpot_ms_p90": pct(0.90),
        "tpot_ms_std": statistics.pstdev(step_ms) if n > 1 else 0.0,
        "tpot_ms_min": min(step_ms),
        "tpot_ms_max": max(step_ms),
    }


def build_server_args(args, n_input_tokens: int) -> ServerArgs:
    """Construct ServerArgs for one attention mode.

    Mirrors profile_decode.py's dataclass-backfill so it is robust to
    ServerArgs fields that this sglang build does not expose via CLI.
    """
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns, _ = parser.parse_known_args([])
    for field in dataclasses.fields(ServerArgs):
        if hasattr(ns, field.name):
            continue
        if field.default is not dataclasses.MISSING:
            setattr(ns, field.name, field.default)
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
            setattr(ns, field.name, field.default_factory())  # type: ignore[misc]
        else:
            setattr(ns, field.name, None)
    ns.model_path = args.model_path
    server_args = ServerArgs.from_cli_args(ns)

    is_quest = args.attention == "quest"
    # headroom for input + warmup + measured decode tokens
    max_seq = max(args.max_seq_lens, n_input_tokens + args.warmup_steps + args.measured_steps + 64)

    # --- backend + scheduling -------------------------------------------
    server_args.model_path = args.model_path
    server_args.attention_backend = "flashinfer"
    server_args.disable_overlap_schedule = True
    server_args.disable_cuda_graph = False
    server_args.disable_radix_cache = True            # fair: no KV-cache reuse
    server_args.tp_size = 1
    server_args.mem_fraction_static = args.mem_fraction_static
    server_args.context_length = max_seq
    server_args.kv_cache_dtype = "auto"               # bf16 KV
    try:
        server_args.cuda_graph_max_bs = max(args.batch_sizes)
    except Exception:
        pass

    # --- vortex / quest --------------------------------------------------
    server_args.enable_vortex_sparsity = is_quest
    server_args.vortex_module_name = QUEST_MODULE
    server_args.vortex_block_size = 16
    server_args.page_size = 16
    server_args.vortex_topk_val = args.topk_val
    server_args.vortex_topk_ratio = 0.0               # pure static budget
    server_args.vortex_block_reserved_bos = 1
    server_args.vortex_block_reserved_eos = 2
    server_args.vortex_workload_chunk_size = 32
    server_args.vortex_layers_skip = [0]
    server_args.vortex_schedule_policy = DEFAULT_SCHEDULE_POLICY
    server_args.vortex_dtype = "bfloat16"
    server_args.vortex_max_seq_lens = max_seq
    server_args.vortex_compilation_cache_dir = args.vortex_cache_dir
    return server_args


def make_reqs(input_ids: List[int], batch_size: int, max_new_tokens: int) -> List[Req]:
    """Replicate the request `batch_size` times as prefix-free Req objects."""
    sampling_params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)
    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []                       # no radix-cache prefix sharing
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)
    return reqs


def time_decode(next_token_ids, batch, model_runner, warmup: int, measured: int) -> List[float]:
    """Run warmup + measured decode steps; return per-step latency in ms.

    Each latency is the CUDA-event interval between consecutive decode()
    calls, so it includes host-side launch overhead in addition to GPU
    kernel time. This overhead is identical for the quest and dense modes,
    so the relative TPOT comparison -- the benchmark's actual deliverable --
    is unaffected.
    """
    for _ in range(warmup):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
    torch.cuda.synchronize()

    events = [torch.cuda.Event(enable_timing=True) for _ in range(measured + 1)]
    events[0].record()
    for i in range(measured):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        events[i + 1].record()
    torch.cuda.synchronize()
    return [events[i].elapsed_time(events[i + 1]) for i in range(measured)]


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def run(args) -> None:
    raw_path = Path(args.raw_csv)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not raw_path.exists() or raw_path.stat().st_size == 0

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    input_ids = build_input_ids(args.request, tokenizer)
    n_input = len(input_ids)
    print(f"[setup] attention={args.attention}  input_tokens={n_input}")

    server_args = build_server_args(args, n_input)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    _set_envs_and_config(server_args)
    configure_logger(server_args, prefix=" TP0")
    port_args = PortArgs.init_new(server_args)
    model_runner, _ = load_model(server_args, port_args, tp_rank=0)

    backend = getattr(model_runner, "attn_backend", None)
    backend_cls = f"{type(backend).__module__}.{type(backend).__name__}" if backend else "<none>"
    print(f"[setup] resolved attention backend: {backend_cls}")
    print(f"[setup] enable_vortex_sparsity={server_args.enable_vortex_sparsity}")

    max_new_tokens = args.warmup_steps + args.measured_steps + 8
    run_ts = datetime.now(timezone.utc).isoformat()
    model_tag = Path(args.model_path).name

    with raw_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
        if write_header:
            writer.writeheader()

        def emit(batch_size, step_idx, latency_ms, status):
            writer.writerow({
                "run_timestamp": run_ts,
                "attention": args.attention,
                "batch_size": batch_size,
                "model": model_tag,
                "topk_val": args.topk_val if args.attention == "quest" else "",
                "input_tokens": n_input,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.measured_steps,
                "step_idx": step_idx,
                "step_latency_ms": "" if latency_ms is None else f"{latency_ms:.6f}",
                "status": status,
            })

        for batch_size in sorted(args.batch_sizes):
            print(f"[run] batch_size={batch_size} ...", flush=True)
            try:
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                reqs = make_reqs(input_ids, batch_size, max_new_tokens)
                with torch.no_grad():
                    next_token_ids, _, batch = extend(reqs, model_runner)
                    torch.cuda.synchronize()
                    step_ms = time_decode(
                        next_token_ids, batch, model_runner,
                        args.warmup_steps, args.measured_steps,
                    )
                for step_idx, latency in enumerate(step_ms):
                    emit(batch_size, step_idx, latency, "ok")
                f.flush()
                stats = summarize_step_latencies(step_ms)
                print(f"[run] batch_size={batch_size}  TPOT mean={stats['tpot_ms_mean']:.3f} ms "
                      f"p50={stats['tpot_ms_p50']:.3f}  p90={stats['tpot_ms_p90']:.3f}")
            except Exception as exc:  # noqa: BLE001
                if _is_oom(exc):
                    print(f"[run] batch_size={batch_size} OOM -- recording N/A, stopping sweep")
                    emit(batch_size, -1, None, "oom")
                    f.flush()
                    torch.cuda.empty_cache()
                    # stopping the ascending sweep keeps any partially-allocated
                    # KV-pool state from the failed extend() harmless: a larger
                    # batch would only OOM harder, and continuing past an OOM
                    # would risk running on corrupt pool state.
                    break
                emit(batch_size, -1, None, "error")
                f.flush()
                raise

    print(f"[done] raw rows written to {raw_path}")


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Quest decode-speed (TPOT) batch benchmark")
    p.add_argument("--attention", choices=["quest", "dense"], required=True)
    p.add_argument("--model-path",
                   default="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-8B")
    p.add_argument("--request", default=str(here / "request.json"))
    p.add_argument("--raw-csv", default=str(here / "results" / "raw_results.csv"))
    p.add_argument("--batch-sizes", type=lambda s: [int(x) for x in s.split(",")],
                   default=DEFAULT_BATCH_SIZES)
    p.add_argument("--topk-val", type=int, default=64,
                   help="Quest static block budget (blocks of 16 tokens kept).")
    p.add_argument("--warmup-steps", type=int, default=16)
    p.add_argument("--measured-steps", type=int, default=128)
    p.add_argument("--max-seq-lens", type=int, default=16384)
    p.add_argument("--mem-fraction-static", type=float, default=0.9)
    p.add_argument("--vortex-cache-dir", default=str(here / ".vortex_cache"))
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
