#!/usr/bin/env python3
"""Quest decode-speed (TPOT) benchmark across decode batch sizes.

Boots ONE `sgl.Engine` for a single attention mode (`quest` or `dense`) and,
for each batch size, measures decode TPOT the same way the project's sgl
baseline does (`sparse_attn/sglang_log/measure_batch_latency_offline.py`, run
as `run_batch_experiments_offline.sh tpot-no-share`):

  * replicate the benchmark request `batch_size` times,
  * warm up the batch shape with one untimed generate,
  * `engine.generate(..., stream=True)` and time the stream,
  * TPOT = (end - first_token_time) / (tokens_generated - 1).

The engine config mirrors the `tpot-no-share` baseline: CUDA graph disabled,
radix cache disabled, flashinfer backend, debug logging. One CSV row is written
per (batch size, repeat). A batch whose KV footprint exceeds the engine's KV
pool is run by sglang in waves; its rows are flagged status=capped but still
recorded -- the baseline reports those points too.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# --- runtime environment for JIT compilation -------------------------------
# flashinfer and vortex (the quest kernels) JIT-compile CUDA code at runtime:
# they need `ninja` on PATH and a CUDA toolkit reachable via CUDA_HOME.
# sgl-kernel 0.3.17 also probes CUDA_HOME at import time. These must be set
# BEFORE torch / sglang / sgl-kernel / flashinfer / vortex import.
_CUDA_HOME = "/vast/parcc/spack/sw/apps/linux-sapphirerapids/cuda-12.8.1-lmm74gnqr2pl2dzbtfjdwoo3fnwbar43"
os.environ.setdefault("CUDA_HOME", _CUDA_HOME)
# sglang 0.5.9 hard-checks the CuDNN version against torch at startup and
# refuses to boot; the stack's bundled CuDNN 9.10 trips that check on
# torch 2.9.1. The smoke gate confirmed decode produces coherent output (a
# sanity check, not a CuDNN-matched numerical validation) -- sufficient for
# a TPOT timing benchmark. Skip the over-strict check.
os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
os.environ["PATH"] = os.pathsep.join([
    str(Path(__file__).resolve().parent / ".venv" / "bin"),
    os.path.join(os.environ["CUDA_HOME"], "bin"),
    os.environ.get("PATH", ""),
])

import vortex_torch  # noqa: F401,E402  -- registers the vortex attention backend
from vortex_torch.engine.sgl import DEFAULT_SCHEDULE_POLICY  # noqa: E402

import sglang as sgl  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from prompt_io import build_prompt_text  # noqa: E402

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
QUEST_MODULE = "gqa_quest_sparse_attention"

# decode steps for the per-batch warmup -- enough to compile/plan every kernel
# the timed repeats reuse (flashinfer plans are shape-dependent, not step-count
# dependent), without paying a full 256-token generate per batch size.
_WARMUP_TOKENS = 16

RAW_CSV_FIELDS = [
    "run_timestamp", "attention", "batch_size", "model", "topk_val",
    "input_tokens", "max_tokens", "repeat_idx", "tokens_generated",
    "ttft_ms", "tpot_ms", "decode_time_ms", "total_time_ms",
    "throughput_tok_s", "status",
]

# metric columns emitted to the raw CSV (besides the descriptor columns)
_METRIC_COLS = ("tokens_generated", "ttft_ms", "tpot_ms", "decode_time_ms",
                "total_time_ms", "throughput_tok_s")


def build_engine_kwargs(args, n_input_tokens: int) -> dict:
    """sgl.Engine kwargs for one attention mode on vortex_torch v0.5.

    Mirrors the `tpot-no-share` sgl baseline (CUDA graph off, radix cache off,
    flashinfer backend, debug logging) and disables chunked prefill (offline
    batch inference prefills each request whole). `quest` additionally enables
    v0.5's vortex sparsity with the built-in `gqa_quest_sparse_attention` flow.

    v0.5 notes: `page_size` must be a multiple of `vortex_block_size` (the
    vortex backend asserts this), so it is set to 16 for both modes; the
    vortex backend is present in-process either way once `vortex_torch` is
    imported. `enable_vortex_sparsity=False` makes `dense` full attention.
    """
    # vortex buffers are sized for the input + the decode tokens + headroom
    max_seq = max(args.max_seq_lens, n_input_tokens + args.max_tokens + 64)

    # Disable chunked prefill -- offline batch inference prefills each request
    # whole, never split or co-packed. sglang's -1 "disable" sentinel is
    # rejected by the vortex page-size assertion (chunked_prefill_size %
    # page_size == 0), so set the budget to one whole request rounded up to
    # the 16-token page plus a one-page margin (>= one request, < two).
    chunked_prefill_size = ((n_input_tokens + 15) // 16) * 16 + 16

    kwargs = {
        "model_path": args.model_path,
        "tp_size": 1,
        "trust_remote_code": True,
        "attention_backend": "flashinfer",
        "page_size": 16,
        "kv_cache_dtype": "auto",
        "disable_cuda_graph": not args.enable_cuda_graph,
        "disable_radix_cache": True,
        # The vortex backend reuses shared metadata buffers across forwards;
        # sglang's overlap scheduler runs forwards in a separate thread and
        # races them. Disabling the overlap schedule serializes scheduling and
        # the forward, closing the race. Required for the vortex backend in
        # v0.5 (vortex_torch/engine/sgl/api.py hardcodes the same).
        "disable_overlap_schedule": True,
        "chunked_prefill_size": chunked_prefill_size,
        "decode_log_interval": 1,
        "show_time_cost": True,
        "log_level": "debug",
    }
    if args.mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = args.mem_fraction_static

    if args.attention == "quest":
        kwargs.update({
            "enable_vortex_sparsity": True,
            "vortex_module_name": QUEST_MODULE,
            "vortex_attention_backend": "flashinfer",
            "vortex_topk_val": args.topk_val,
            "vortex_topk_ratio": 0.0,            # pure static block budget
            "vortex_block_size": 16,
            "vortex_block_reserved_bos": 1,
            "vortex_block_reserved_eos": 2,
            "vortex_workload_chunk_size": 32,
            "vortex_layers_skip": [0],
            "vortex_schedule_policy": DEFAULT_SCHEDULE_POLICY,
            "vortex_dtype": "bfloat16",
            "vortex_max_seq_lens": max_seq,
            "vortex_compilation_cache_dir": args.vortex_cache_dir,
        })
    else:
        # dense -- full attention; sparsity explicitly off
        kwargs["enable_vortex_sparsity"] = False
    return kwargs


def build_get_engine_kwargs(args, n_input_tokens: int) -> dict:
    """Kwargs for `vortex_torch.engine.sgl.get_engine` -- Quest's official
    in-process engine constructor.

    `get_engine` hardcodes a few defaults that the baseline `tpot-no-share`
    config flips (notably `disable_cuda_graph=False`, `enable_vortex_sparsity=
    True`). It also accepts `**kwargs` and applies them *after* its own
    defaults, so we override every fairness-relevant flag explicitly here.
    The only meaningful difference between the engine produced by this path
    and the engine produced by `sgl.Engine(**build_engine_kwargs(...))` is
    the constructor call itself -- same model, same backend, same sparsity
    flow, same CUDA-graph / radix-cache / chunked-prefill / overlap-schedule
    posture, same debug logging.
    """
    max_seq = max(args.max_seq_lens, n_input_tokens + args.max_tokens + 64)
    chunked_prefill_size = ((n_input_tokens + 15) // 16) * 16 + 16

    kwargs = dict(
        # get_engine's named params we want to pin
        model_path=args.model_path,
        vortex_max_seq_lens=max_seq,
        vortex_block_size=16,
        vortex_topk_val=args.topk_val,
        vortex_block_reserved_bos=1,
        vortex_block_reserved_eos=2,
        vortex_workload_chunk_size=32,
        vortex_layers_skip=[0],
        vortex_module_name=QUEST_MODULE,
        # gqa_quest_sparse_attention is built-in to vortex_torch v0.5, so the
        # flow is already in the registry by the time get_engine runs;
        # vortex_module_path is only consulted if the name is unregistered.
        # Pointing at this benchmark file (which never @register's anything)
        # is harmless and avoids hardcoding submissions/-relative paths.
        vortex_module_path=str(Path(__file__).resolve()),
        kv_cache_dtype="auto",
        # Fairness overrides -- passed via **kwargs tail in get_engine; they
        # land in the final sgl.Engine kwargs dict via engine_kwargs.update.
        disable_cuda_graph=not args.enable_cuda_graph,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        attention_backend="flashinfer",
        page_size=16,
        chunked_prefill_size=chunked_prefill_size,
        decode_log_interval=1,
        show_time_cost=True,
        log_level="debug",
        trust_remote_code=True,
        vortex_attention_backend="flashinfer",
        vortex_compilation_cache_dir=args.vortex_cache_dir,
        # Explicit: get_engine's hardcoded default is True, but we restate it
        # so the override-every-fairness-flag contract is visible in one place.
        enable_vortex_sparsity=True,
    )
    if args.mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = args.mem_fraction_static

    if args.attention == "dense":
        # get_engine hardcodes enable_vortex_sparsity=True; flip it off for
        # the dense baseline. The vortex_* kwargs remain in the call but are
        # not consulted by sglang when sparsity is off (verified by the
        # smoke test in Task 5).
        kwargs["enable_vortex_sparsity"] = False
    return kwargs


def compute_metrics(start_time: float, first_token_time: Optional[float],
                    end_time: float, tokens_generated: int) -> dict:
    """Turn streaming timestamps into TTFT / decode-time / TPOT (all ms).

    TPOT = decode_time / (tokens_generated - 1): the `-1` drops the first token,
    which streams out at the end of prefill. Identical formula to the baseline's
    `measure_batch_latency`.
    """
    total_time = end_time - start_time
    if first_token_time is None or tokens_generated <= 0:
        return {
            "ttft_ms": total_time * 1000.0,
            "decode_time_ms": 0.0,
            "total_time_ms": total_time * 1000.0,
            "tpot_ms": 0.0,
            "tokens_generated": tokens_generated,
        }
    ttft = first_token_time - start_time
    decode_time = end_time - first_token_time
    tpot = decode_time / (tokens_generated - 1) if tokens_generated > 1 else 0.0
    return {
        "ttft_ms": ttft * 1000.0,
        "decode_time_ms": decode_time * 1000.0,
        "total_time_ms": total_time * 1000.0,
        "tpot_ms": tpot * 1000.0,
        "tokens_generated": tokens_generated,
    }


def kv_pool_capacity(engine) -> Optional[int]:
    """Best-effort KV-pool token capacity the engine reported at boot.

    `sgl.Engine` stores the scheduler handshake info on `.scheduler_info`;
    `max_total_num_tokens` is the count of KV token-slots. Returns None if the
    attribute is absent -- the harness then simply does not flag capped batches.
    """
    info = getattr(engine, "scheduler_info", None)
    if isinstance(info, dict):
        for key in ("max_total_num_tokens", "max_total_tokens"):
            val = info.get(key)
            if val:
                return int(val)
    return None


def classify_status(tokens_generated: int, batch_size: int, seq_len: int,
                    pool_tokens: Optional[int]) -> str:
    """Status for one measured repeat: ok / capped / error.

    `capped` -- the batch's KV footprint (batch_size * seq_len) exceeds the KV
    pool, so sglang ran it in waves; the TPOT is real but at a reduced effective
    batch. `error` -- no decode step was measured (TPOT undefined).
    """
    if tokens_generated <= 1:
        return "error"
    if pool_tokens is not None and batch_size * seq_len > pool_tokens:
        return "capped"
    return "ok"


def _chunk_token_count(item: dict) -> int:
    """Output-token count from a streamed sglang chunk (schema varies by version)."""
    ids = item.get("output_ids")
    if ids is not None:
        return len(ids)
    meta = item.get("meta_info") or {}
    return int(meta.get("completion_tokens", 0))


def warmup_batch(engine, prompts: List[str]) -> None:
    """Run one untimed generate to compile/plan kernels for this batch shape.

    Without this, the first of the timed repeats pays flashinfer's kernel-plan /
    JIT first-call cost -- a one-off spike that corrupts that repeat's TPOT. A
    short warmup (prefill plus a few decode steps) compiles every kernel the
    timed repeats then reuse, so all `repeat` measurements capture steady state.
    """
    engine.generate(
        prompts,
        sampling_params={
            "max_new_tokens": _WARMUP_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
        },
    )


def measure_batch_latency(engine, prompts: List[str], max_tokens: int) -> dict:
    """Measure one batch via streaming generate -- mirrors the sgl baseline.

    Returns the dict from `compute_metrics` plus `throughput_tok_s` (every
    request's tokens over the decode window).
    """
    sampling_params = {
        "max_new_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
    }

    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    tokens_generated = 0

    for chunk in engine.generate(prompts, sampling_params=sampling_params,
                                 stream=True):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        # every request shares max_new_tokens + ignore_eos, so the batch
        # decodes in lockstep -- request 0's running count represents the batch.
        item = chunk[0] if isinstance(chunk, list) else chunk
        tokens_generated = _chunk_token_count(item)

    end_time = time.perf_counter()

    metrics = compute_metrics(start_time, first_token_time, end_time,
                              tokens_generated)
    decode_s = metrics["decode_time_ms"] / 1000.0
    metrics["throughput_tok_s"] = (
        len(prompts) * tokens_generated / decode_s if decode_s > 0 else 0.0
    )
    return metrics


def make_engine(args, n_input_tokens: int):
    """Construct an sgl.Engine via the chosen engine API.

    `direct`     -- current path: build kwargs in this file, call sgl.Engine.
    `get_engine` -- Quest's official wrapper at vortex_torch.engine.sgl.api;
                    it folds in vortex defaults and ends in sgl.Engine(**...).
    The two paths are configured to produce equivalent fairness-relevant
    kwargs (see build_get_engine_kwargs); any TPOT delta between them is
    attributable to the constructor call itself.
    """
    if args.engine_api == "direct":
        return sgl.Engine(**build_engine_kwargs(args, n_input_tokens))
    if args.engine_api == "get_engine":
        from vortex_torch.engine.sgl.api import get_engine
        return get_engine(**build_get_engine_kwargs(args, n_input_tokens))
    raise ValueError(f"unknown engine_api: {args.engine_api!r}")


def run(args) -> None:
    raw_path = Path(args.raw_csv)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not raw_path.exists() or raw_path.stat().st_size == 0

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    prompt = build_prompt_text(args.request, tokenizer)
    n_input = len(tokenizer(prompt, return_tensors=None)["input_ids"])
    seq_len = n_input + args.max_tokens
    print(f"[setup] attention={args.attention}  input_tokens={n_input}", flush=True)

    engine = make_engine(args, n_input)
    try:
        pool_tokens = kv_pool_capacity(engine)
        print(f"[setup] kv pool tokens={pool_tokens}  per-request seq_len={seq_len}",
              flush=True)
        if pool_tokens:
            print(f"[setup] ~{pool_tokens // seq_len} requests fit concurrently; "
                  f"larger batches run in waves (status=capped)", flush=True)

        run_ts = datetime.now(timezone.utc).isoformat()
        model_tag = Path(args.model_path).name

        with raw_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
            if write_header:
                writer.writeheader()

            def emit(batch_size, repeat_idx, metrics, status):
                row = {
                    "run_timestamp": run_ts,
                    "attention": args.attention,
                    "batch_size": batch_size,
                    "model": model_tag,
                    "topk_val": args.topk_val if args.attention == "quest" else "",
                    "input_tokens": n_input,
                    "max_tokens": args.max_tokens,
                    "repeat_idx": repeat_idx,
                    "status": status,
                }
                for col in _METRIC_COLS:
                    v = None if metrics is None else metrics.get(col)
                    if v is None:
                        row[col] = ""
                    elif isinstance(v, float):
                        row[col] = f"{v:.6f}"
                    else:
                        row[col] = v
                writer.writerow(row)

            for batch_size in sorted(args.batch_sizes):
                print(f"[run] batch_size={batch_size} ...", flush=True)
                prompts = [prompt] * batch_size
                try:
                    warmup_batch(engine, prompts)
                    tpots = []
                    for repeat_idx in range(args.repeat):
                        metrics = measure_batch_latency(engine, prompts,
                                                        args.max_tokens)
                        status = classify_status(metrics["tokens_generated"],
                                                 batch_size, seq_len, pool_tokens)
                        emit(batch_size, repeat_idx, metrics, status)
                        f.flush()
                        if status != "error":
                            tpots.append(metrics["tpot_ms"])
                        print(f"[run]   repeat={repeat_idx} status={status} "
                              f"tpot={metrics['tpot_ms']:.3f} ms "
                              f"ttft={metrics['ttft_ms']:.1f} ms", flush=True)
                    if tpots:
                        mean = sum(tpots) / len(tpots)
                        print(f"[run] batch_size={batch_size}  "
                              f"TPOT mean={mean:.3f} ms", flush=True)
                except Exception:  # noqa: BLE001
                    emit(batch_size, -1, None, "error")
                    f.flush()
                    raise

        print(f"[done] raw rows written to {raw_path}", flush=True)
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Quest decode-speed (TPOT) batch benchmark")
    p.add_argument("--attention", choices=["quest", "dense"], required=True)
    p.add_argument("--engine-api", choices=["direct", "get_engine"],
                   default="direct",
                   help="Engine constructor path. 'direct' (current default) "
                        "calls sgl.Engine(**build_engine_kwargs); 'get_engine' "
                        "routes through vortex_torch.engine.sgl.get_engine "
                        "with identical fairness flags. Used by the "
                        "engine-API comparison sweep.")
    p.add_argument("--model-path",
                   default="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct",
                   help="Headline model is Qwen3-VL-8B-Instruct (the sgl "
                        "baseline's model). Fallback: .../Qwen/Qwen3-8B if "
                        "Qwen3-VL cannot run -- see README.md.")
    p.add_argument("--request", default=str(here / "request.json"))
    p.add_argument("--raw-csv", default=str(here / "results" / "raw_results.csv"))
    p.add_argument("--batch-sizes", type=lambda s: [int(x) for x in s.split(",")],
                   default=DEFAULT_BATCH_SIZES)
    p.add_argument("--topk-val", type=int, default=64,
                   help="Quest static block budget (blocks of 16 tokens kept).")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Output tokens generated per request (matches the baseline).")
    p.add_argument("--repeat", type=int, default=3,
                   help="Measurements per batch size (matches the baseline).")
    p.add_argument("--mem-fraction-static", type=float, default=None,
                   help="Static model+KV memory fraction. Default None = the "
                        "sglang default, matching the baseline; the engine "
                        "chunks prefill so a large batch wave-serializes "
                        "rather than OOMs.")
    p.add_argument("--enable-cuda-graph", action="store_true",
                   help="Run decode with a CUDA graph. Off by default to match "
                        "the baseline, which disables CUDA graph for timing.")
    p.add_argument("--max-seq-lens", type=int, default=16384,
                   help="Lower bound for vortex buffer sizing (quest only).")
    p.add_argument("--vortex-cache-dir", default=str(here / ".vortex_cache"))
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
