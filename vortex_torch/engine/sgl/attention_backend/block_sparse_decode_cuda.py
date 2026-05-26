"""Option A — minimal CUDA port of the Triton v1 block-sparse decode kernel.

Identical math to ``_block_sparse_decode_kernel_v1`` in
``block_sparse_decode_triton.py``, written in CUDA + JIT-compiled via
``torch.utils.cpp_extension.load_inline``. Provides a clear migration
path off Triton if we later need finer control over shared memory,
tensor cores, or sm-specific tuning.

Kernel layout:
  - one CUDA block per (CSR-row n, query head g)
  - BLOCK_D=head_dim threads per CUDA block (BLOCK_D=128 in this branch)
  - online softmax state in shared memory: q_smem[D], k_smem[BS][D],
    v_smem[BS][D], acc_smem[D], scores_smem[BS], plus scalars m_i, l_i, alpha
  - inner loop walks block_ids one at a time (no tile fusion in v1)

Enable with ``VORTEX_CUSTOM_KERNEL_VERSION=cuda`` alongside
``VORTEX_USE_CUSTOM=1``.

Build is JIT — first call compiles, subsequent calls reuse the cached .so
under ``~/.cache/torch_extensions``. If the build fails (e.g. on sm_120
with a broken toolchain), the wrapper returns ``None`` and the caller
falls back to Triton v1.
"""
from __future__ import annotations

import sys
from typing import Optional

import torch


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

// Block-sparse decode v1, mirroring the Triton implementation 1:1.
//   - one CUDA block per (row n, query head g)
//   - BLOCK_D threads per CUDA block (one per head-dim lane)
//   - fp32 accumulation, bf16 I/O
template <int BS, int BLOCK_D>
__global__ void block_sparse_decode_kernel_v1_cuda(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const int32_t* __restrict__ INDPTR,
    const int32_t* __restrict__ INDICES,
    const int32_t* __restrict__ LAST_LEN,
    __nv_bfloat16* __restrict__ O,
    float sm_scale,
    int stride_qn, int stride_qg, int stride_qd,
    int stride_kb, int stride_kt, int stride_kd,
    int stride_vb, int stride_vt, int stride_vd,
    int stride_on, int stride_og, int stride_od
) {
    const int n = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ float q_smem[BLOCK_D];
    __shared__ float k_smem[BS][BLOCK_D];
    __shared__ float v_smem[BS][BLOCK_D];
    __shared__ float acc_smem[BLOCK_D];
    __shared__ float scores_smem[BS];
    __shared__ float m_i_smem;
    __shared__ float l_i_smem;
    __shared__ float alpha_smem;

    // ---- Init Q in smem + acc=0, m=-inf, l=0 ----
    if (tid < BLOCK_D) {
        q_smem[tid] = __bfloat162float(
            Q[n * stride_qn + g * stride_qg + tid * stride_qd]);
        acc_smem[tid] = 0.0f;
    }
    if (tid == 0) {
        m_i_smem = -FLT_MAX;
        l_i_smem = 0.0f;
    }
    __syncthreads();

    const int row_start = INDPTR[n];
    const int row_end = INDPTR[n + 1];
    const int last_len = LAST_LEN[n];

    for (int i = row_start; i < row_end; ++i) {
        const int block_id = INDICES[i];

        // ---- Load K[block_id, :, :] and V[block_id, :, :] ----
        if (tid < BLOCK_D) {
            #pragma unroll
            for (int t = 0; t < BS; ++t) {
                k_smem[t][tid] = __bfloat162float(
                    K[block_id * stride_kb + t * stride_kt + tid * stride_kd]);
                v_smem[t][tid] = __bfloat162float(
                    V[block_id * stride_vb + t * stride_vt + tid * stride_vd]);
            }
        }
        __syncthreads();

        // ---- Compute scores[t] = q . k[t]   (BS small → 1 thread per t) ----
        if (tid < BS) {
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < BLOCK_D; ++d) {
                s += q_smem[d] * k_smem[tid][d];
            }
            scores_smem[tid] = s * sm_scale;
        }
        __syncthreads();

        // ---- Online softmax update (single-threaded; BS is tiny) ----
        if (tid == 0) {
            const bool is_last = (i == row_end - 1);
            const float m_old = m_i_smem;
            const float l_old = l_i_smem;

            // Apply last-block validity mask + compute m_new
            float m_new = m_old;
            #pragma unroll
            for (int t = 0; t < BS; ++t) {
                float s = scores_smem[t];
                bool valid = (!is_last) || (t < last_len);
                if (!valid) s = -FLT_MAX;
                scores_smem[t] = s;
                if (s > m_new) m_new = s;
            }
            float alpha = expf(m_old - m_new);

            // Build P[t] = exp(s - m_new), update l
            float l_new = l_old * alpha;
            #pragma unroll
            for (int t = 0; t < BS; ++t) {
                float p = expf(scores_smem[t] - m_new);
                scores_smem[t] = p;  // overwrite with normalized p
                l_new += p;
            }

            m_i_smem = m_new;
            l_i_smem = l_new;
            alpha_smem = alpha;
        }
        __syncthreads();

        // ---- Parallel acc rescale + accumulate p·v ----
        if (tid < BLOCK_D) {
            float a = acc_smem[tid] * alpha_smem;
            #pragma unroll
            for (int t = 0; t < BS; ++t) {
                a += scores_smem[t] * v_smem[t][tid];
            }
            acc_smem[tid] = a;
        }
        __syncthreads();
    }

    // ---- Write O[n, g, :] = acc / l ----
    if (tid < BLOCK_D) {
        float o_val = acc_smem[tid] / l_i_smem;
        O[n * stride_on + g * stride_og + tid * stride_od] = __float2bfloat16(o_val);
    }
}


at::Tensor block_sparse_decode_v1_cuda_launch(
    at::Tensor q, at::Tensor cache_k, at::Tensor cache_v,
    at::Tensor indptr, at::Tensor indices, at::Tensor last_block_len,
    double sm_scale)
{
    TORCH_CHECK(q.is_cuda() && cache_k.is_cuda() && cache_v.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q.dtype() == at::kBFloat16, "q must be bf16");
    TORCH_CHECK(cache_k.dtype() == at::kBFloat16, "cache_k must be bf16");
    TORCH_CHECK(cache_v.dtype() == at::kBFloat16, "cache_v must be bf16");
    TORCH_CHECK(indptr.dtype() == at::kInt, "indptr must be int32");
    TORCH_CHECK(indices.dtype() == at::kInt, "indices must be int32");
    TORCH_CHECK(last_block_len.dtype() == at::kInt, "last_block_len must be int32");

    const int N = q.size(0);
    const int G = q.size(1);
    const int D = q.size(2);
    const int BS = cache_k.size(1);

    TORCH_CHECK(BS == 4, "v1 CUDA only supports BS=4");
    TORCH_CHECK(D == 128 || D == 64, "v1 CUDA supports head_dim 64 or 128");

    auto out = at::empty_like(q);
    dim3 grid(N, G);

    if (D == 128) {
        dim3 block(128);
        block_sparse_decode_kernel_v1_cuda<4, 128><<<grid, block>>>(
            reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(cache_k.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(cache_v.data_ptr()),
            indptr.data_ptr<int32_t>(),
            indices.data_ptr<int32_t>(),
            last_block_len.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<float>(sm_scale),
            q.stride(0), q.stride(1), q.stride(2),
            cache_k.stride(0), cache_k.stride(1), cache_k.stride(2),
            cache_v.stride(0), cache_v.stride(1), cache_v.stride(2),
            out.stride(0), out.stride(1), out.stride(2));
    } else {  // D == 64
        dim3 block(64);
        block_sparse_decode_kernel_v1_cuda<4, 64><<<grid, block>>>(
            reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(cache_k.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(cache_v.data_ptr()),
            indptr.data_ptr<int32_t>(),
            indices.data_ptr<int32_t>(),
            last_block_len.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            static_cast<float>(sm_scale),
            q.stride(0), q.stride(1), q.stride(2),
            cache_k.stride(0), cache_k.stride(1), cache_k.stride(2),
            cache_v.stride(0), cache_v.stride(1), cache_v.stride(2),
            out.stride(0), out.stride(1), out.stride(2));
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
"""


_module: Optional[object] = None
_BUILD_FAILED = False


def _try_build():
    """JIT-compile the CUDA source. Returns the module or None on failure."""
    global _module, _BUILD_FAILED
    if _module is not None:
        return _module
    if _BUILD_FAILED:
        return None
    try:
        from torch.utils.cpp_extension import load_inline
        # Forward declaration so the auto-generated pybind module
        # (compiled from cpp_sources) can find the symbol defined in cuda_sources.
        _cpp_header = (
            "#include <torch/extension.h>\n"
            "at::Tensor block_sparse_decode_v1_cuda_launch("
            "  at::Tensor q, at::Tensor cache_k, at::Tensor cache_v,"
            "  at::Tensor indptr, at::Tensor indices, at::Tensor last_block_len,"
            "  double sm_scale);\n"
        )
        _module = load_inline(
            name="block_sparse_decode_v1_cuda",
            cpp_sources=_cpp_header,
            cuda_sources=_CUDA_SRC,
            functions=["block_sparse_decode_v1_cuda_launch"],
            verbose=False,
            extra_cuda_cflags=["-O2", "--use_fast_math"],
        )
        # Expose under the short name "run" for symmetry with Triton.
        _module.run = _module.block_sparse_decode_v1_cuda_launch
        return _module
    except Exception as e:
        print(
            f"[VORTEX] CUDA kernel build failed: {type(e).__name__}: {e}\n"
            "         Falling back to Triton v1.",
            file=sys.stderr,
        )
        _BUILD_FAILED = True
        return None


class BlockSparseDecodeCUDA:
    """Thin wrapper. ``run(...)`` matches the Triton path's signature."""

    def run(self, q, cache_k, cache_v, indptr, indices, last_block_len, sm_scale):
        mod = _try_build()
        if mod is None:
            return None
        return mod.run(q, cache_k, cache_v, indptr, indices, last_block_len, sm_scale)
