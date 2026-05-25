"""Block-size sweep — Variant D: block_size=1 (token-level, 16x finer).

Same algorithm as id0; only the JSON config differs.

NOTE: block_size=1 means page_size is also forced to 1 (vortex_torch sets
page_size=block_size at engine init, see vortex_torch/engine/sgl/api.py:48).
This is the most aggressive setting — FlashInfer's BatchDecode kernel was
not designed for page_size=1, so expect either (a) very high overhead from
per-token indirect indexing, or (b) runtime failure. mem_fraction_static
lowered to 0.70 to leave room for the much larger per-token page-table.
"""
import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, GeMM, Mean
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("block_size_sweep_id3_cls")
class BlockSizeSweepId3Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.gemm = GeMM()
        self.mean = Mean(dim=1)
        self.output_func = topK()
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = self.mean(q, ctx=ctx)
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
        }
