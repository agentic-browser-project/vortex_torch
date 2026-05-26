"""HBM bytes-read accounting for the sparse-attention path.

Each decoder step calls `record(indptr_last, block_size, head_dim, bytes_per_elem)`
once per layer, telling us how many block_ids ended up in
``sparse_kv_indices``. We tally them and at process exit dump a summary
to ``$VORTEX_HBM_TRACE``.

Why this works: a "block" is the unit the kernel actually loads — its
size is ``block_size * head_dim * sizeof(bf16)`` for K and the same
again for V. Total bytes the kernel must pull from HBM per layer per
step ≈ ``total_indices * block_size * head_dim * 2 (K+V) * bytes_per_elem``,
ignoring metadata + indptr overhead (a few hundred bytes per row,
negligible at our scales).

Enabled by ``VORTEX_HBM_TRACE=<path>`` (a file path); a no-op otherwise.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch

_TOTAL_INDICES = 0
_TOTAL_CALLS = 0
_TOTAL_BYTES = 0
_ENABLED: Optional[bool] = None
_OUT_PATH: Optional[str] = None
# Flush every N calls to avoid losing trace on SIGKILL of the scheduler subprocess.
# A typical RULER run is ~1500 layer×step calls, so flushing every 32 calls
# writes ~50 times — negligible compared to GPU work.
_FLUSH_EVERY = 32


def _maybe_init() -> bool:
    global _ENABLED, _OUT_PATH
    if _ENABLED is None:
        _OUT_PATH = os.environ.get("VORTEX_HBM_TRACE")
        if _OUT_PATH:
            # Resolve relative paths against the launching CWD (env var captures
            # that, since the env is inherited by the scheduler subprocess).
            _OUT_PATH = os.path.abspath(_OUT_PATH)
        _ENABLED = bool(_OUT_PATH)
    return _ENABLED


def record(indptr: torch.Tensor, n_rows: int, block_size: int,
           head_dim: int, bytes_per_elem: int = 2) -> None:
    """Tally one layer×step and periodically flush the running totals to disk."""
    if not _maybe_init():
        return
    global _TOTAL_INDICES, _TOTAL_CALLS, _TOTAL_BYTES
    total = int(indptr[n_rows].item())
    _TOTAL_INDICES += total
    _TOTAL_CALLS += 1
    _TOTAL_BYTES += total * block_size * head_dim * 2 * bytes_per_elem  # 2x for K+V
    if _TOTAL_CALLS % _FLUSH_EVERY == 0:
        _dump()


def _dump() -> None:
    if not _ENABLED or _OUT_PATH is None:
        return
    payload = {
        "total_indices_summed": _TOTAL_INDICES,
        "total_calls": _TOTAL_CALLS,
        "total_kv_bytes_read": _TOTAL_BYTES,
        "avg_indices_per_call": (
            _TOTAL_INDICES / _TOTAL_CALLS if _TOTAL_CALLS else 0
        ),
        "avg_kv_mb_per_call": (
            _TOTAL_BYTES / _TOTAL_CALLS / 1e6 if _TOTAL_CALLS else 0
        ),
        "total_kv_mb": _TOTAL_BYTES / 1e6,
        "policy": os.environ.get("VORTEX_POLICY", "(unset)"),
        "use_bsr": os.environ.get("VORTEX_USE_BSR", "0"),
        "use_custom": os.environ.get("VORTEX_USE_CUSTOM", "0"),
        "kernel_version": os.environ.get("VORTEX_CUSTOM_KERNEL_VERSION", "(unset)"),
    }
    try:
        os.makedirs(os.path.dirname(_OUT_PATH) or ".", exist_ok=True)
        # Write to a tempfile + rename for atomicity, so partial flushes
        # don't leave a half-written JSON on disk.
        tmp_path = _OUT_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, _OUT_PATH)
    except Exception as e:
        print(f"[VORTEX_HBM_TRACE] failed to write {_OUT_PATH}: {e}")
