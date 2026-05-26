"""Standalone Triton smoke test for the Option A block-sparse decode kernel.

Compares the Triton kernel's output against a naive PyTorch reference on a
tiny synthetic problem (N=2, G=4, BS=4, D=64, num_blocks=16). Passes if
all positions match to within bfloat16 round-off (~1e-2 absolute).

Run:
    python algorithm_scientist/test_custom_kernel.py
"""
import importlib.util
import os
import sys

import torch

# Load the kernel module directly to avoid pulling in flashinfer.py and the
# whole sglang import chain (which requires env vars like SGLANG_ENABLE_TORCH_COMPILE).
_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vortex_torch/engine/sgl/attention_backend/block_sparse_decode_triton.py",
)
_spec = importlib.util.spec_from_file_location("block_sparse_decode_triton", _KERNEL_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["block_sparse_decode_triton"] = _mod
_spec.loader.exec_module(_mod)
BlockSparseDecodeTriton = _mod.BlockSparseDecodeTriton


def naive_reference(q, cache_k, cache_v, indptr, indices, last_len, sm_scale):
    """Same math as the kernel, in PyTorch."""
    N, G, D = q.shape
    BS = cache_k.shape[1]
    out = torch.zeros_like(q)
    for n in range(N):
        start = int(indptr[n].item())
        end = int(indptr[n + 1].item())
        ll = int(last_len[n].item())
        sel = indices[start:end]  # [num_sel]
        k_gather = cache_k[sel].reshape(-1, D)  # [num_sel * BS, D]
        v_gather = cache_v[sel].reshape(-1, D)
        # mask: last `BS - ll` positions of the LAST block are invalid
        valid = torch.ones(k_gather.shape[0], dtype=torch.bool, device=q.device)
        if ll < BS and len(sel) > 0:
            invalid_start = (len(sel) - 1) * BS + ll
            valid[invalid_start:] = False
        for g in range(G):
            qg = q[n, g].float()                              # [D]
            scores = (qg @ k_gather.float().T) * sm_scale     # [num_sel*BS]
            scores = scores.masked_fill(~valid, float("-inf"))
            p = torch.softmax(scores, dim=0)
            out[n, g] = (p @ v_gather.float()).to(out.dtype)
    return out


def test_basic():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    N, G, BS, D = 2, 4, 4, 64
    num_blocks = 16

    q = torch.randn(N, G, D, device=device, dtype=dtype)
    cache_k = torch.randn(num_blocks, BS, D, device=device, dtype=dtype)
    cache_v = torch.randn(num_blocks, BS, D, device=device, dtype=dtype)

    # row 0: blocks [3, 7, 11]; row 1: blocks [0, 1, 5, 9]
    indices = torch.tensor([3, 7, 11, 0, 1, 5, 9], device=device, dtype=torch.int32)
    indptr = torch.tensor([0, 3, 7], device=device, dtype=torch.int32)
    last_len = torch.tensor([BS, 2], device=device, dtype=torch.int32)  # row 1 last block has 2 valid tokens
    sm_scale = 1.0 / (D ** 0.5)

    kernel = BlockSparseDecodeTriton()
    o_tri = kernel.run(q, cache_k, cache_v, indptr, indices, last_len, sm_scale)
    o_ref = naive_reference(q, cache_k, cache_v, indptr, indices, last_len, sm_scale)

    diff = (o_tri.float() - o_ref.float()).abs()
    print(f"o_tri shape: {tuple(o_tri.shape)}")
    print(f"o_ref shape: {tuple(o_ref.shape)}")
    print(f"max abs diff:  {diff.max().item():.6f}")
    print(f"mean abs diff: {diff.mean().item():.6f}")
    print(f"o_tri  mean={o_tri.float().mean().item():.4f} std={o_tri.float().std().item():.4f}")
    print(f"o_ref  mean={o_ref.float().mean().item():.4f} std={o_ref.float().std().item():.4f}")

    # bfloat16 tolerance — soft on max abs diff, strict on mean
    assert diff.max().item() < 5e-2, f"max diff too large: {diff.max().item()}"
    assert diff.mean().item() < 1e-2, f"mean diff too large: {diff.mean().item()}"
    print("PASS")


if __name__ == "__main__":
    test_basic()
