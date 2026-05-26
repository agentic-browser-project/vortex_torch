"""Option A — custom Triton block-sparse decode kernel.

Goal: do flash-attention decode where K/V are gathered at block-granularity
(block_size=4 tokens at a time) directly inside the attention kernel, with
no paging indirection. This is the kernel form of "true sub-page fetch":
HBM loads happen at block_size granularity, set by the sparse_kv_indices
buffer.

Contract (mirrors the BSR wrapper integration in flashinfer.py):
  q              : [N, G, D]              bfloat16
  cache_k        : [total_blocks, BS, D]   bfloat16  (num_kv_heads=1 in the
                                                      vortex_torch layout)
  cache_v        : [total_blocks, BS, D]   bfloat16
  indptr         : [N+1]                  int32, CSR row pointers
  indices        : [indptr[-1]]           int32, physical block ids
  last_block_len : [N]                    int32, valid tokens in the last
                                          block of each row (∈ [1, BS])
  out (returned) : [N, G, D]              bfloat16

N = batch_size * num_kv_heads. G = group_size (Q heads per KV head).
BS = block_size (4 in this branch). D = head_dim (128 for Qwen3).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _block_sparse_decode_kernel(
    Q_ptr,                   # [N, G, D]
    K_ptr,                   # [total_blocks, BS, D]
    V_ptr,                   # [total_blocks, BS, D]
    INDPTR_ptr,              # [N+1]
    INDICES_ptr,             # [num_total_indices]
    LAST_LEN_ptr,            # [N]
    O_ptr,                   # [N, G, D]
    sm_scale,
    G,
    stride_qn, stride_qg, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_on, stride_og, stride_od,
    BS: tl.constexpr,        # block_size (number of tokens per gather block)
    BLOCK_D: tl.constexpr,   # head_dim (power of 2)
):
    """One program per (row n, query head g).

    Each program walks its CSR row, doing flash-attention online softmax over
    the gathered K/V blocks.
    """
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    d_off = tl.arange(0, BLOCK_D)              # [D]
    t_off = tl.arange(0, BS)                   # [BS]

    # Load Q[n, g, :] in fp32 for accumulation accuracy.
    q = tl.load(
        Q_ptr + pid_n * stride_qn + pid_g * stride_qg + d_off * stride_qd
    ).to(tl.float32)

    # Online softmax state.
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    row_start = tl.load(INDPTR_ptr + pid_n).to(tl.int32)
    row_end = tl.load(INDPTR_ptr + pid_n + 1).to(tl.int32)
    last_len = tl.load(LAST_LEN_ptr + pid_n).to(tl.int32)

    # Iterate blocks selected for this row. One block per iter; later we can
    # tile this further if BS=4 turns out too small for warp utilisation.
    for i in range(row_start, row_end):
        block_id = tl.load(INDICES_ptr + i).to(tl.int32)

        # Gather K[block_id], V[block_id] of shape [BS, D].
        k_tile = tl.load(
            K_ptr
            + block_id * stride_kb
            + t_off[:, None] * stride_kt
            + d_off[None, :] * stride_kd
        ).to(tl.float32)
        v_tile = tl.load(
            V_ptr
            + block_id * stride_vb
            + t_off[:, None] * stride_vt
            + d_off[None, :] * stride_vd
        ).to(tl.float32)

        # scores = q · k_tile.T → [BS]
        scores = tl.sum(q[None, :] * k_tile, axis=1) * sm_scale

        # Mask out invalid tokens in the LAST block only. We can't trivially
        # tell from inside the kernel which iteration is the last one, so we
        # apply a conservative mask on every block — but since non-last
        # blocks always have last_len == BS effectively, we instead mask
        # only when i == row_end-1.
        is_last = (i == row_end - 1)
        valid_mask = (~is_last) | (t_off < last_len)
        scores = tl.where(valid_mask, scores, float("-inf"))

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        # Re-zero the masked positions in p so they don't contribute (exp of
        # -inf is 0 but safer to be explicit).
        p = tl.where(valid_mask, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        m_i = m_new

    o = acc / l_i

    tl.store(
        O_ptr + pid_n * stride_on + pid_g * stride_og + d_off * stride_od,
        o.to(O_ptr.dtype.element_ty),
    )


class BlockSparseDecodeTriton:
    """Wrapper that mirrors the .run() interface of FlashInfer's wrappers.

    No .plan() needed — Triton autotunes at first call; all metadata travels
    in the run-time arguments.
    """

    def __init__(self):
        pass

    def run(
        self,
        q: torch.Tensor,           # [N, G, D]  bfloat16
        cache_k: torch.Tensor,     # [num_blocks, BS, D]  bfloat16
        cache_v: torch.Tensor,     # [num_blocks, BS, D]  bfloat16
        indptr: torch.Tensor,      # [N+1]  int32
        indices: torch.Tensor,     # [indptr[-1]]  int32
        last_block_len: torch.Tensor,  # [N]  int32
        sm_scale: float,
    ) -> torch.Tensor:
        assert q.dim() == 3 and cache_k.dim() == 3 and cache_v.dim() == 3
        N, G, D = q.shape
        num_blocks, BS, D_k = cache_k.shape
        assert D == D_k, f"head_dim mismatch: q={D} k={D_k}"
        assert indptr.shape == (N + 1,)
        assert last_block_len.shape == (N,)

        # head_dim must be power of 2 for Triton's tl.arange.
        assert (D & (D - 1)) == 0, f"head_dim={D} must be power of 2"

        out = torch.empty_like(q)

        grid = (N, G)
        _block_sparse_decode_kernel[grid](
            q, cache_k, cache_v, indptr, indices, last_block_len, out,
            float(sm_scale),
            G,
            q.stride(0), q.stride(1), q.stride(2),
            cache_k.stride(0), cache_k.stride(1), cache_k.stride(2),
            cache_v.stride(0), cache_v.stride(1), cache_v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BS=BS,
            BLOCK_D=D,
            num_warps=4,
            num_stages=2,
        )
        return out
