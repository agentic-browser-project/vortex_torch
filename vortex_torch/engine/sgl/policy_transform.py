"""Policy transformation hook for vortex_torch's sparse attention.

After vortex_torch's indexer fills in sparse_kv_indices (one block_id per
4-token chunk selected), this module rewrites the indices/indptr in place
to simulate one of three fetch policies:

  - block_fetch          : passthrough (vortex_torch's current behavior)
  - method1_p<P>         : if any selected block is in a P-page, load ALL
                           blocks of that P-page (lossless over-fetch)
  - method2_p<P>_t<TT>   : load a P-page only when its hit ratio >= TT/100,
                           drop the rest entirely (lossy)

P is the conceptual "page" size for Method 1/2 in tokens (must be a
multiple of block_size). TT in {10, 25, 50, 75}.

Triggered by environment variable VORTEX_POLICY. Examples:

    VORTEX_POLICY=block_fetch
    VORTEX_POLICY=method1_p16
    VORTEX_POLICY=method2_p32_t50

block_id encoding (from planner_sglang.py):
    block_id = page_id * num_blocks_per_page + (token_pos % page_size) / block_size
    page_id  = (token_pos / page_size) * num_kv_heads + kv_head
So for a kv_head h's CSR row, block_ids within the row come from this
formula, all with the same kv_head = h.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Optional, Tuple

import torch

_POLICY_RE = re.compile(r"^(block_fetch|method1_p(\d+)|method2_p(\d+)_t(\d+))$")


def parse_policy(policy_str: str) -> Optional[Tuple[str, int, float]]:
    """Returns (method, P, threshold) or None for block_fetch/empty.

    Raises ValueError for unrecognised strings.
    """
    if not policy_str or policy_str == "block_fetch":
        return None
    m = _POLICY_RE.match(policy_str)
    if not m:
        raise ValueError(f"Unrecognized VORTEX_POLICY: {policy_str!r}")
    if m.group(2):  # method1_p<P>
        return ("method1", int(m.group(2)), 0.0)
    else:           # method2_p<P>_t<TT>
        return ("method2", int(m.group(3)), int(m.group(4)) / 100.0)


def _logical_block_idx(physical_block: int, kv_head: int,
                       num_kv_heads: int, num_blocks_per_page: int) -> int:
    """Invert planner formula: physical block_id -> logical block index within
    this kv_head's sequence (0-indexed, contiguous integers)."""
    q, r = divmod(physical_block, num_blocks_per_page)
    s = (q - kv_head) // num_kv_heads
    return s * num_blocks_per_page + r


def _physical_block_id(logical_block: int, kv_head: int,
                       num_kv_heads: int, num_blocks_per_page: int) -> int:
    """Forward: logical block index -> physical block_id."""
    s, r = divmod(logical_block, num_blocks_per_page)
    q = s * num_kv_heads + kv_head
    return q * num_blocks_per_page + r


def apply_policy(
    indptr: torch.Tensor,         # int32 [n_rows+1], CSR row pointers (on-device)
    indices_buf: torch.Tensor,    # int32 [max_total_blocks], the buffer itself
    num_kv_heads: int,
    block_size: int,
    num_blocks_per_page: int,
    policy_str: str,
    n_rows: Optional[int] = None, # Active rows (bs * num_kv_heads). Buffers are
                                  # pre-allocated for the max batch, so without
                                  # this we'd treat trailing zeros as valid rows.
) -> None:
    """Modify indptr and indices_buf IN PLACE according to policy_str.

    No-op for block_fetch / empty / None policy_str.
    On buffer overflow (Method 1 expansion exceeds capacity), prints a
    warning and leaves the buffers unchanged (effectively block_fetch).
    """
    parsed = parse_policy(policy_str)
    if parsed is None:
        return

    method, P, threshold = parsed
    if P % block_size != 0:
        raise ValueError(f"P={P} must be a multiple of block_size={block_size}")
    blocks_per_p_page = P // block_size

    indptr_cpu = indptr.cpu().tolist()
    if n_rows is None:
        n_rows = len(indptr_cpu) - 1
    total_len = int(indptr_cpu[n_rows])
    if total_len == 0:
        return

    indices_cpu = indices_buf[:total_len].cpu().tolist()
    buf_capacity = indices_buf.numel()

    new_per_row: List[List[int]] = []
    for row_idx in range(n_rows):
        start, end = int(indptr_cpu[row_idx]), int(indptr_cpu[row_idx + 1])
        if start == end:
            new_per_row.append([])
            continue

        kv_head = row_idx % num_kv_heads
        original_blocks = indices_cpu[start:end]

        # physical -> logical
        logical_blocks = [
            _logical_block_idx(b, kv_head, num_kv_heads, num_blocks_per_page)
            for b in original_blocks
        ]

        # Group by P-page
        p_page_hits: dict[int, int] = defaultdict(int)
        for lb in logical_blocks:
            p_page_hits[lb // blocks_per_p_page] += 1

        # Select P-pages to load (Method 1 / Method 2)
        if method == "method1":
            kept_p_pages = list(p_page_hits.keys())
        else:  # method2
            min_hits = max(1, int(threshold * blocks_per_p_page + 0.999999))
            kept_p_pages = [p for p, c in p_page_hits.items() if c >= min_hits]
            # Fallback: avoid empty selection (would crash FlashInfer)
            if not kept_p_pages and logical_blocks:
                kept_p_pages = [logical_blocks[0] // blocks_per_p_page]

        # Expand kept P-pages back to all-block lists (logical, then physical)
        expanded_logical = []
        for p in kept_p_pages:
            for i in range(blocks_per_p_page):
                expanded_logical.append(p * blocks_per_p_page + i)
        expanded_logical = sorted(set(expanded_logical))

        new_physical = [
            _physical_block_id(lb, kv_head, num_kv_heads, num_blocks_per_page)
            for lb in expanded_logical
        ]
        new_per_row.append(sorted(new_physical))

    # Check buffer capacity (Method 1 can grow indices ~blocks_per_p_page X)
    new_total = sum(len(r) for r in new_per_row)
    if new_total > buf_capacity:
        # Soft fail: leave buffers alone, behavior degrades to block_fetch
        print(f"[VORTEX_POLICY={policy_str}] buffer overflow "
              f"({new_total} > {buf_capacity}), falling back to block_fetch")
        return

    # Coverage / waste / page-hit-histogram accounting (VORTEX_POLICY_STATS=1).
    #   coverage = |selected ∩ loaded| / |selected|   (information preserved)
    #   waste    = |loaded - selected| / |loaded|     (over-fetch ratio)
    # Aggregated across all rows in this layer × step into a global tally
    # dumped to the path in VORTEX_HBM_TRACE.
    import os as _os
    if _os.environ.get("VORTEX_POLICY_STATS"):
        from vortex_torch.engine.sgl.hbm_trace import record_policy_stats, record_page_hist
        n_sel = n_loaded = n_kept = 0
        page_hits_dist = [0] * (blocks_per_p_page + 1)
        for row_idx in range(n_rows):
            start, end = int(indptr_cpu[row_idx]), int(indptr_cpu[row_idx + 1])
            pre = set(indices_cpu[start:end])
            post = set(new_per_row[row_idx])
            n_sel += len(pre)
            n_loaded += len(post)
            n_kept += len(pre & post)
            # Per-row page-hit count — clamp to valid bucket in case of dupes
            kv_head = row_idx % num_kv_heads
            local_page_hits = defaultdict(int)
            for b in indices_cpu[start:end]:
                lb = _logical_block_idx(b, kv_head, num_kv_heads, num_blocks_per_page)
                local_page_hits[lb // blocks_per_p_page] += 1
            for cnt in local_page_hits.values():
                page_hits_dist[min(cnt, blocks_per_p_page)] += 1
        record_policy_stats(n_sel, n_loaded, n_kept)
        record_page_hist(page_hits_dist)

    # Reconstruct flat indices + cumulative indptr
    new_indices_flat: List[int] = []
    new_indptr = [0]
    for row in new_per_row:
        new_indices_flat.extend(row)
        new_indptr.append(new_indptr[-1] + len(row))

    if not new_indices_flat:
        # All rows empty (shouldn't happen given fallback above, but defend)
        return

    # Write back (in place)
    new_indices_t = torch.tensor(
        new_indices_flat, dtype=indices_buf.dtype, device=indices_buf.device
    )
    indices_buf[:new_indices_t.numel()] = new_indices_t

    new_indptr_t = torch.tensor(
        new_indptr, dtype=indptr.dtype, device=indptr.device
    )
    indptr[:new_indptr_t.numel()] = new_indptr_t
