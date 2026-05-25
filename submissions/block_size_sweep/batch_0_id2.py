"""Block-size sweep — Variant C: block_size=4 (4x finer).

Same algorithm as id0; only the JSON config differs.
"""
import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, GeMM, Mean
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("block_size_sweep_id2_cls")
class BlockSizeSweepId2Cls(vFlow):
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
