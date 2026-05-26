"""Option A — custom Triton block-sparse decode kernel.

Goal: do flash-attention decode where K/V are gathered at block-granularity
(block_size=4 tokens at a time) directly inside the attention kernel, with
no paging indirection. HBM loads happen at block_size granularity, set by
the sparse_kv_indices buffer.

Two kernels are exposed:

  - v1 ``_block_sparse_decode_kernel`` — one program per (row, q_head),
    processes one block per inner iteration. Simple, easy to reason about.
  - v2 ``_block_sparse_decode_kernel_v2`` — one program per row,
    processes ``BLOCK_BLOCKS`` blocks per inner iteration AND fuses all
    G GQA-group query heads together so the QK matmul becomes
    ``(G, D) @ (D, BLOCK_BLOCKS*BS)`` — a real ``tl.dot`` that hits
    tensor cores. autotune'd over (BLOCK_BLOCKS, num_warps, num_stages).

Contract:
  q              : [N, G, D]              bfloat16
  cache_k        : [num_blocks, BS, D]    bfloat16  (num_kv_heads=1 in the
                                                     vortex_torch layout)
  cache_v        : [num_blocks, BS, D]    bfloat16
  indptr         : [N+1]                  int32, CSR row pointers
  indices        : [indptr[-1]]           int32, physical block ids
  last_block_len : [N]                    int32, valid tokens in the last
                                          block of each row (∈ [1, BS])
  out (returned) : [N, G, D]              bfloat16

N = batch_size * num_kv_heads. G = group_size (Q heads per KV head).
BS = block_size (4 in this branch). D = head_dim (128 for Qwen3).

Select between v1 and v2 by setting ``VORTEX_CUSTOM_KERNEL_VERSION`` to
``"v1"``, ``"v2"``, or ``"auto"`` (default: ``"auto"``).

Auto-selection: v2 is faster ONLY when ``G >= 8`` because it pads the M
dim of the QK matmul to 16 (tl.dot minimum). For small G (e.g. Qwen3
with G=2), v2 wastes ~8× compute and ``auto`` falls back to v1.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


# =============================================================================
# v1 — one program per (row, q_head), one block per iter.
# =============================================================================


@triton.jit
def _block_sparse_decode_kernel_v1(
    Q_ptr, K_ptr, V_ptr,
    INDPTR_ptr, INDICES_ptr, LAST_LEN_ptr,
    O_ptr,
    sm_scale,
    G,
    stride_qn, stride_qg, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_on, stride_og, stride_od,
    BS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    d_off = tl.arange(0, BLOCK_D)
    t_off = tl.arange(0, BS)

    q = tl.load(
        Q_ptr + pid_n * stride_qn + pid_g * stride_qg + d_off * stride_qd
    ).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    row_start = tl.load(INDPTR_ptr + pid_n).to(tl.int32)
    row_end = tl.load(INDPTR_ptr + pid_n + 1).to(tl.int32)
    last_len = tl.load(LAST_LEN_ptr + pid_n).to(tl.int32)

    for i in range(row_start, row_end):
        block_id = tl.load(INDICES_ptr + i).to(tl.int32)

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

        scores = tl.sum(q[None, :] * k_tile, axis=1) * sm_scale

        is_last = (i == row_end - 1)
        valid_mask = (~is_last) | (t_off < last_len)
        scores = tl.where(valid_mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(valid_mask, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        m_i = m_new

    o = acc / l_i

    tl.store(
        O_ptr + pid_n * stride_on + pid_g * stride_og + d_off * stride_od,
        o.to(O_ptr.dtype.element_ty),
    )


# =============================================================================
# v2 — one program per row, BLOCK_BLOCKS blocks per iter, GQA-fused
#      QK and PV via tl.dot. Hits tensor cores when (G_PAD, D) and
#      (D, BLOCK_BLOCKS*BS) meet the MMA minimum shapes.
# =============================================================================
#
# Design notes:
#   - One program per row n. Processes ALL G query heads in the matmul's
#     M dimension. To survive tl.dot's MMA minimum (16x16), we PAD G to
#     G_PAD = max(G, 16) and mask the padded heads with -inf in QK.
#   - Inner loop iterates over chunks of BLOCK_BLOCKS blocks. Each chunk
#     contributes BLOCK_BLOCKS * BS tokens to the K dimension.
#   - Online softmax maintains (m_i, l_i, acc) at shape [G_PAD], with
#     acc being [G_PAD, D].
#   - Gathered K/V tiles are loaded as bfloat16 to feed tl.dot in fp16/bf16
#     and accumulate in fp32 (Triton's default for tl.dot).


# tl.dot requires M, N, K >= 16. With BS=4, BLOCK_BLOCKS*BS >= 16 → BLOCK_BLOCKS >= 4.
_V2_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_BLOCKS": bb}, num_warps=nw, num_stages=ns)
    for bb in [4, 8, 16]
    for nw in [2, 4, 8]
    for ns in [1, 2, 3]
]


@triton.autotune(
    configs=_V2_AUTOTUNE_CONFIGS,
    key=["BS", "BLOCK_D", "G_PAD"],
)
@triton.jit
def _block_sparse_decode_kernel_v2(
    Q_ptr, K_ptr, V_ptr,
    INDPTR_ptr, INDICES_ptr, LAST_LEN_ptr,
    O_ptr,
    sm_scale,
    G,
    stride_qn, stride_qg, stride_qd,
    stride_kb, stride_kt, stride_kd,
    stride_vb, stride_vt, stride_vd,
    stride_on, stride_og, stride_od,
    BS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    G_PAD: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
):
    pid_n = tl.program_id(0)

    d_off = tl.arange(0, BLOCK_D)                      # [D]
    g_off = tl.arange(0, G_PAD)                        # [G_PAD]
    g_valid = g_off < G                                 # [G_PAD]

    # ---- Load Q[n, :G, :] padded to [G_PAD, D] -----------------------------
    q_ptrs = Q_ptr + pid_n * stride_qn \
        + g_off[:, None] * stride_qg + d_off[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=g_valid[:, None], other=0.0)  # [G_PAD, D] bf16

    # ---- Init online softmax state ----------------------------------------
    m_i = tl.full([G_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([G_PAD], dtype=tl.float32)
    acc = tl.zeros([G_PAD, BLOCK_D], dtype=tl.float32)

    row_start = tl.load(INDPTR_ptr + pid_n).to(tl.int32)
    row_end = tl.load(INDPTR_ptr + pid_n + 1).to(tl.int32)
    last_len = tl.load(LAST_LEN_ptr + pid_n).to(tl.int32)
    row_nblocks = row_end - row_start                  # int32

    # ---- Iterate in BLOCK_BLOCKS chunks ------------------------------------
    bb_off = tl.arange(0, BLOCK_BLOCKS)                # [BLOCK_BLOCKS]
    t_off = tl.arange(0, BS)                           # [BS]
    # Token offset within a chunk: [BLOCK_BLOCKS * BS]
    chunk_t = (bb_off[:, None] * BS + t_off[None, :]).reshape(BLOCK_BLOCKS * BS)

    # Number of chunks (round up). The tail chunk is masked.
    num_chunks = (row_nblocks + BLOCK_BLOCKS - 1) // BLOCK_BLOCKS

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * BLOCK_BLOCKS         # int32

        # ---- Per-token validity mask within this chunk --------------------
        bb_global = chunk_start + bb_off               # [BLOCK_BLOCKS]
        bb_in_row = bb_global < row_nblocks             # [BLOCK_BLOCKS]
        is_last_block = bb_global == (row_nblocks - 1)  # [BLOCK_BLOCKS]
        # Per-token: block-in-row AND (not-last-block OR within last_len)
        token_valid_2d = bb_in_row[:, None] & (
            (~is_last_block[:, None]) | (t_off[None, :] < last_len)
        )                                               # [BLOCK_BLOCKS, BS]
        token_valid = token_valid_2d.reshape(BLOCK_BLOCKS * BS)  # [B*BS]

        # ---- Block ID lookup (clamped to a safe value for masked-out blocks)
        # For OOB blocks we read index 0 with a mask=False load; the result
        # is dropped via token_valid in the softmax anyway.
        idx_ptrs = INDICES_ptr + row_start + bb_global
        block_ids = tl.load(idx_ptrs, mask=bb_in_row, other=0).to(tl.int32)

        # ---- Gather K, V into [BLOCK_BLOCKS*BS, D] -------------------------
        # k_ptrs[i, t, d] = K_ptr + block_ids[i]*stride_kb + t*stride_kt + d*stride_kd
        # Flattened along (i, t) to [BLOCK_BLOCKS*BS, D].
        block_ids_flat = (
            tl.broadcast_to(block_ids[:, None], [BLOCK_BLOCKS, BS])
            .reshape(BLOCK_BLOCKS * BS)
        )
        k_ptrs = K_ptr \
            + block_ids_flat[:, None] * stride_kb \
            + (chunk_t % BS)[:, None] * stride_kt \
            + d_off[None, :] * stride_kd
        v_ptrs = V_ptr \
            + block_ids_flat[:, None] * stride_vb \
            + (chunk_t % BS)[:, None] * stride_vt \
            + d_off[None, :] * stride_vd

        k_tile = tl.load(k_ptrs, mask=token_valid[:, None], other=0.0)  # [B*BS, D]
        v_tile = tl.load(v_ptrs, mask=token_valid[:, None], other=0.0)  # [B*BS, D]

        # ---- QK matmul: [G_PAD, D] @ [D, B*BS] -> [G_PAD, B*BS] ------------
        # tl.dot wants (M, K) @ (K, N). k_tile is [B*BS, D], we want its
        # transpose [D, B*BS]. Triton supports `tl.dot(a, b)` with b's K
        # dim implicit via shape.
        scores = tl.dot(q, tl.trans(k_tile)).to(tl.float32) * sm_scale  # [G_PAD, B*BS]

        # Mask invalid columns AND invalid query rows
        mask_2d = g_valid[:, None] & token_valid[None, :]
        scores = tl.where(mask_2d, scores, float("-inf"))

        # ---- Online softmax update ----------------------------------------
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))            # [G_PAD]
        alpha = tl.exp(m_i - m_new)                                 # [G_PAD]
        p = tl.exp(scores - m_new[:, None])                         # [G_PAD, B*BS]
        p = tl.where(mask_2d, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)                       # [G_PAD]

        # acc = acc * alpha + p @ v_tile
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile).to(tl.float32)
        m_i = m_new

    # ---- Final normalization + store --------------------------------------
    o = acc / l_i[:, None]                                          # [G_PAD, D]
    o_ptrs = O_ptr + pid_n * stride_on \
        + g_off[:, None] * stride_og + d_off[None, :] * stride_od
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=g_valid[:, None])


# =============================================================================
# Python wrapper
# =============================================================================


def _next_pow2(x: int) -> int:
    n = 1
    while n < x:
        n <<= 1
    return n


class BlockSparseDecodeTriton:
    """Wrapper that mirrors the .run() interface of FlashInfer's wrappers.

    Pick the kernel version with ``VORTEX_CUSTOM_KERNEL_VERSION`` env var
    (``"v1"`` or ``"v2"``; default ``"v2"``). v2 uses tile fusion +
    autotune + tensor-core matmul; v1 is the simpler reference impl.
    """

    def __init__(self):
        pass

    def run(
        self,
        q: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_block_len: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        assert q.dim() == 3 and cache_k.dim() == 3 and cache_v.dim() == 3
        N, G, D = q.shape
        num_blocks, BS, D_k = cache_k.shape
        assert D == D_k
        assert indptr.shape == (N + 1,)
        assert last_block_len.shape == (N,)
        assert (D & (D - 1)) == 0, f"head_dim={D} must be power of 2"

        out = torch.empty_like(q)

        version = os.environ.get("VORTEX_CUSTOM_KERNEL_VERSION", "auto")
        if version == "auto":
            version = "v2" if G >= 8 else "v1"
        if version == "cuda":
            # Option A — hand-written CUDA port of v1. Falls back to Triton
            # v1 if the JIT build fails (e.g. broken toolchain on sm_120).
            # Load via importlib to avoid pulling in the full sglang package
            # graph when this file is imported from a smoke test.
            if not hasattr(self, "_cuda_wrapper"):
                import importlib.util as _ilu
                _cuda_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "block_sparse_decode_cuda.py",
                )
                _spec = _ilu.spec_from_file_location(
                    "block_sparse_decode_cuda", _cuda_path
                )
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                self._cuda_wrapper = _mod.BlockSparseDecodeCUDA()
            out_cuda = self._cuda_wrapper.run(
                q, cache_k, cache_v, indptr, indices, last_block_len, sm_scale,
            )
            if out_cuda is not None:
                return out_cuda
            # build failure → fall through to Triton v1
            version = "v1"
        if version == "v1":
            grid = (N, G)
            _block_sparse_decode_kernel_v1[grid](
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
        else:
            # v2: pad G to at least 16 for tensor-core friendliness.
            G_PAD = max(_next_pow2(G), 16)
            grid = (N,)
            _block_sparse_decode_kernel_v2[grid](
                q, cache_k, cache_v, indptr, indices, last_block_len, out,
                float(sm_scale),
                G,
                q.stride(0), q.stride(1), q.stride(2),
                cache_k.stride(0), cache_k.stride(1), cache_k.stride(2),
                cache_v.stride(0), cache_v.stride(1), cache_v.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BS=BS,
                BLOCK_D=D,
                G_PAD=G_PAD,
            )
        return out
