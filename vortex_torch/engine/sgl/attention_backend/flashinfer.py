from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Dict, Tuple
from functools import partial
import torch
from vortex_torch import is_hopper
from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.indexer import Context, MetaData
from vortex_torch.indexer.compiler.compile import compile as compile_indexer
from vortex_torch.indexer.utils_sglang import (
    get_chunkwise_hn2nh_transpose,
    get_chunkwise_nh2hn_transpose,
    get_decode_planner,
    get_prefill_planner,
)
if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.layers.attention.flashinfer_backend import should_use_tensor_core
if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        BlockSparseAttentionWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import _get_range_buf, get_seq_lens

# PATCH (Option A): custom Triton block-sparse decode kernel. Lives in a
# sibling module so it can be developed/tested independently of FlashInfer.
from vortex_torch.engine.sgl.attention_backend.block_sparse_decode_triton import (
    BlockSparseDecodeTriton,
)

@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]

@dataclass
class PrefillMetadata:
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class VortexFlashInferBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        assert model_runner.sliding_window_size is None
        assert not model_runner.model_config.is_encoder_decoder 
        assert not self.skip_prefill
        assert not self.is_multimodal
        assert kv_indptr_buf is None
        assert kv_last_page_len_buf is None
        self.num_wrappers = 2
        self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        # if (
        #     "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
        #     or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
        #     or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
        # ):
        #     global_config.flashinfer_workspace_size = 512 * 1024 * 1024

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer
        max_bs = model_runner.req_to_token_pool.size
        
        self.num_qo_heads = model_runner.model_config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(get_attention_tp_size())
        self.group_size = self.num_qo_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.decode_use_tensor_cores = should_use_tensor_core(self.data_type, self.num_qo_heads, self.num_kv_heads)
        assert self.q_data_type in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
        assert self.data_type in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
        self.is_fp8 = (self.data_type in [torch.float8_e5m2, torch.float8_e4m3fn])
        
        # Assign key configuration and parameters
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.server_args.page_size
        self.block_size = model_runner.server_args.vortex_block_size
        self.layers_skip = model_runner.server_args.vortex_layers_skip
        self.num_blocks_per_page = self.page_size // self.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size."
        # ===========================
        # Prefill KV-indptr buffers
        # ===========================

        self.kv_indptr_prefill = torch.zeros(
            (max_bs * self.num_kv_heads + 1,),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # Decode-path buffers live on ``self.ctx.metadata`` (pre-allocated
        # in ``_compile``). The flashinfer wrappers consume them directly
        # via ``self.ctx.metadata.dense_kv_indptr`` etc.
        # ===========================

        # ===========================
        # KV indices (prefill) — still owned by this object (the flashinfer
        # prefill wrapper has its own indptr/indices arrays).
        # ===========================

        self.kv_indices_prefill = torch.zeros(
            (
                (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1)
                // self.page_size,
            ),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # KV last-page-len (prefill — decode lives on MetaData)
        # ===========================

        self.kv_last_page_len_prefill = torch.ones(
            (max_bs * self.num_kv_heads,),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # Query/Output indptr buffers
        # ===========================

        self.qo_indptr = [
            torch.zeros(
                (max_bs + 1,),
                dtype=torch.int32,
                device=model_runner.device
            ),
            torch.zeros(
                (max_bs * self.num_kv_heads + 1,),
                dtype=torch.int32,
                device=model_runner.device
            ),
        ]

        # ===========================
        # Batch table (token-level mapping)
        # ===========================

        self.batch_table = torch.zeros(
            (model_runner.server_args.max_prefill_tokens,),
            dtype=torch.uint16,
            device=model_runner.device
        )

        
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend= "auto" if not is_hopper() else "fa3"
        )

        self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2" if ((not is_hopper()) or self.is_fp8) else "fa3",
                    )
        
        self.decode_wrappers = [
            BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                ),
            BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                ),
        ]

        # PATCH (block_size_sweep, Option C): an alternative sparse-attention
        # wrapper that uses BSR (flat KV + arbitrary block gather) instead of
        # paged decode. Activated when env var VORTEX_USE_BSR=1 is set.
        # The BSR path goes through FA2/FA3 prefill kernel (not decode-specialized),
        # so it may be slower at q_len=1 — that's exactly what we're benchmarking.
        self.bsr_wrapper = BlockSparseAttentionWrapper(self.workspace_buffer)

        # PATCH (block_size_sweep, Option A): a custom Triton block-sparse
        # decode kernel that fuses sub-page (block_size=4) gather directly
        # into the attention kernel — no FlashInfer wrapper involved.
        # Activated when env var VORTEX_USE_CUSTOM=1 is set.
        self.custom_decode = BlockSparseDecodeTriton()
        
        self.plan_decode = get_decode_planner(model_runner.server_args.vortex_schedule_policy)
        self.plan_prefill = get_prefill_planner()
        self.chunkwise_nh2hn_transpose = get_chunkwise_nh2hn_transpose()
        self.chunkwise_hn2nh_transpose = get_chunkwise_hn2nh_transpose()

        self.sparse_attention = model_runner.sparse_attention
        self.ctx = Context()
        self._compile(model_runner)
        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata: Dict[int, List[BatchDecodeWithPagedKVCacheWrapper]] = {}
        self.plan_graph: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]]
    

    def _compile(self, model_runner: "ModelRunner") -> None:
        """Trace the sparse-attention indexer on zero-sized dummies and compile it."""
        device = model_runner.device
        dtype = self.q_data_type
        indexer = self.sparse_attention.forward_indexer

        self.ctx.create(self, model_runner)
        # Allocate every per-forward-batch buffer (winfo_*, dense/sparse
        # kv_indptr+indices, kv_last_page_len) on a single MetaData owned
        # by the context. The decode planner writes into this MetaData;
        # the indexer kernels read from it, and the flashinfer decode
        # wrappers below take pointers into it.
        self.ctx.metadata = MetaData.preallocate(self.ctx, device=device)
        self.ctx.assert_created()
        self.ctx.profile()

        def register(vt, name: str) -> None:
            self.ctx.tensor_list.append(vt)
            self.ctx.output_tensor_to_op_list.append(None)
            self.ctx.tensor_id_to_tensor_name_map[vt.tensor_id] = name

        def make_dummy(shape, fmt, tensor_id, *, tdtype=dtype, zeros=False):
            factory = torch.zeros if zeros else torch.empty
            return as_vtensor(factory(shape, device=device, dtype=tdtype), fmt, tensor_id=tensor_id)

        with torch.no_grad():
            q_dummy = make_dummy((0, self.group_size, self.head_dim), FORMAT.BATCHED, tensor_id=0)
            register(q_dummy, "q")

            o_dummy = make_dummy((0, 1, 1), FORMAT.RAGGED, tensor_id=1)
            register(o_dummy, "o")

            cache_dummy = {}
            for i, (name, (shape, cache_dtype)) in enumerate(
                self.sparse_attention.get_cache_meta_info().items()
            ):
                vt = make_dummy(
                    (0, shape[0], shape[1]),
                    FORMAT.PAGED,
                    tensor_id=2 + i,
                    tdtype=cache_dtype,
                    zeros=True,
                )
                cache_dummy[name] = vt
                register(vt, f"cache['{name}']")

            indexer(q_dummy, o_dummy, cache_dummy, ctx=self.ctx)

        self.compiled_indexer = compile_indexer(self.ctx)()
        self.ctx.summary()
        self.ctx.execute()

    
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        
        assert not forward_batch.forward_mode.is_draft_extend()
        assert not forward_batch.forward_mode.is_target_verify()
        
        if forward_batch.forward_mode.is_decode_or_idle():
            
            bs = len(forward_batch.req_pool_indices)
            self.plan_decode(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                ctx=self.ctx
            )
            
            self.decode_wrappers[0].plan(
                indptr=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.dense_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            self.decode_wrappers[1].plan(
                indptr=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.sparse_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )

            # PATCH (Option C): BSR-wrapper plan is NOT called here. Unlike
            # BatchDecodeWithPagedKVCacheWrapper (which captures the indices
            # buffer pointer at __init__ and re-reads at run-time),
            # BlockSparseAttentionWrapper.plan() snapshots indices values at
            # plan time. The indexer fills sparse_kv_indices per-layer inside
            # forward_decode, so we must plan there (right before BSR.run).
            # We pre-record M/N here so forward_decode can plan quickly.
            if os.environ.get("VORTEX_USE_BSR") == "1":
                self._bsr_M = bs * self.num_kv_heads
                self._bsr_N = self.ctx.metadata.sparse_kv_indices.numel() * self.block_size

            self.forward_metadata = DecodeMetadata([self.decode_wrappers[0], self.decode_wrappers[1]])

        elif forward_batch.forward_mode.is_extend():
            
            prefix_lens = forward_batch.extend_prefix_lens
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            bs = len(forward_batch.req_pool_indices)
            
            self.plan_prefill(
                cached_seq_lens=prefix_lens,
                dense_kv_indptr=self.kv_indptr_prefill[:bs*self.num_kv_heads+1],
                dense_kv_indices=self.kv_indices_prefill,
                input_seq_lens=(forward_batch.seq_lens.to(torch.int32) - prefix_lens),
                qo_indptr_ragged=self.qo_indptr[0][:bs+1],
                qo_indptr_paged=self.qo_indptr[1][:bs*self.num_kv_heads+1],
                kv_last_page_len=self.kv_last_page_len_prefill[:bs*self.num_kv_heads],
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                batch_table=self.batch_table,
                page_size=self.page_size,
                num_kv_heads=self.num_kv_heads
            )
            
   
            self.prefill_wrapper_ragged.plan(
                self.qo_indptr[0][:bs+1],
                self.qo_indptr[0][:bs+1],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )
            
            self.prefill_wrapper_paged.plan(
                self.qo_indptr[1][:bs*self.num_kv_heads+1],
                self.kv_indptr_prefill[:bs*self.num_kv_heads+1],
                self.kv_indices_prefill,
                self.kv_last_page_len_prefill[:bs*self.num_kv_heads],
                self.group_size,
                1,
                self.head_dim,
                self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                custom_mask=None,
                non_blocking=True,
            )
            

            self.forward_metadata = PrefillMetadata(extend_no_prefix)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        pass
    
    
    def capture_plan_graph(
        self, 
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int):
        
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
    ):  
        assert bs == num_tokens
        
        if forward_mode.is_decode_or_idle():
            decode_wrappers = [
                BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.ctx.metadata.dense_kv_indices,
                        paged_kv_last_page_len_buffer=self.ctx.metadata.kv_last_page_len[
                            :bs*self.num_kv_heads
                        ],
                    ),
                
                BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.ctx.metadata.sparse_kv_indices,
                        paged_kv_last_page_len_buffer=self.ctx.metadata.kv_last_page_len[
                            :bs*self.num_kv_heads
                        ],
                    ),
                
            ]

            self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )
            
            decode_wrappers[0].plan(
                indptr=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.dense_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            decode_wrappers[1].plan(
                indptr=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.sparse_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)             
        else:
            raise NotImplementedError
            

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        assert forward_mode.is_decode_or_idle()
        
        self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )
        
        self.decode_cuda_graph_metadata[bs][0].plan(
            indptr=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads+1],
            indices=self.ctx.metadata.dense_kv_indices,
            last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
            num_qo_heads=self.group_size,
            num_kv_heads=1,
            head_dim=self.head_dim,
            page_size=self.block_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
        )
        
        self.decode_cuda_graph_metadata[bs][1].plan(
            indptr=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads+1],
            indices=self.ctx.metadata.sparse_kv_indices,
            last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
            num_qo_heads=self.group_size,
            num_kv_heads=1,
            head_dim=self.head_dim,
            page_size=self.block_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc
        
        logits_soft_cap = layer.logit_cap

        q = q.contiguous()

        if self.forward_metadata.extend_no_prefix:
            o = self.prefill_wrapper_ragged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

        else:
            o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                )
            
            q_t = self.chunkwise_nh2hn_transpose(
                q.view(-1, self.num_qo_heads, self.head_dim),
                self.qo_indptr[0],
                self.batch_table,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim
            )
            
            
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            k_cache = k_cache.view(-1, self.page_size, 1, self.head_dim)
            v_cache = v_cache.view(-1, self.page_size, 1, self.head_dim)
            o2, s2 = self.prefill_wrapper_paged.forward_return_lse(
                q_t,
                (k_cache, v_cache),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                )
            o2_t, s2_t = self.chunkwise_hn2nh_transpose(
                o2,  s2,
                self.qo_indptr[0],
                self.batch_table,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim
            )
            
            o, _ = merge_state(o1, s1, o2_t, s2_t)

        if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """
        Decode-time forward pass with optional sparse attention.
        Expects KV to be sourced from token_to_kv_pool; can also save new KV.
        """

        # Sanity checks and setup
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc

        # Optionally write incoming K/V to decode cache
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Read Cache from memory pool
        cache = forward_batch.token_to_kv_pool.get_cache(layer.layer_id)
        
        cache_k = cache["k"].view(-1, self.block_size, 1, self.head_dim)
        cache_v = cache["v"].view(-1, self.block_size, 1, self.head_dim)
        
        # Decide whether to use sparsity on this layer
        use_sparsity = (layer.layer_id not in self.layers_skip)

        if use_sparsity:
            # Prepare Q in grouped shape expected by sparse path
            q = q.contiguous().view(-1, self.group_size, layer.head_dim)

            # Build sparse indices into paged KV buffers
            self.compiled_indexer.forward(
                q=q,
                o=self.forward_metadata.decode_wrappers[1]._paged_kv_indices_buf,
                cache=cache,
                ctx=self.ctx
            )

            # PATCH (block_size_sweep): dump sparse_kv_indices for trace-driven policy analysis
            _dump_dir = os.environ.get("VORTEX_DUMP_TRACE_DIR")
            if _dump_dir:
                import json as _json
                _indptr_full = self.ctx.metadata.sparse_kv_indptr.cpu().tolist()
                # Trim indptr: find the first index where the sequence stops being non-decreasing
                _real_len = len(_indptr_full)
                for _i in range(1, len(_indptr_full)):
                    if _indptr_full[_i] < _indptr_full[_i - 1]:
                        _real_len = _i
                        break
                _indptr = _indptr_full[:_real_len]
                _total = _indptr[-1] if _indptr else 0
                _indices = self.forward_metadata.decode_wrappers[1]._paged_kv_indices_buf[:_total].cpu().tolist()
                _last_page_len = self.ctx.metadata.kv_last_page_len[:len(_indptr)-1].cpu().tolist()
                _trace_path = os.path.join(_dump_dir, f"trace_layer{layer.layer_id}.jsonl")
                os.makedirs(_dump_dir, exist_ok=True)
                with open(_trace_path, "a") as _f:
                    _f.write(_json.dumps({
                        "layer_id": layer.layer_id,
                        "num_kv_heads": self.num_kv_heads,
                        "block_size": self.block_size,
                        "indptr": _indptr,
                        "indices": _indices,
                        "last_page_len": _last_page_len,
                    }) + "\n")

            # PATCH (block_size_sweep): apply Method 1 / Method 2 fetch policy
            # by rewriting sparse_kv_indptr/indices in place before attention.
            _policy = os.environ.get("VORTEX_POLICY")
            if _policy and _policy != "block_fetch":
                from vortex_torch.engine.sgl.policy_transform import apply_policy
                apply_policy(
                    indptr=self.ctx.metadata.sparse_kv_indptr,
                    indices_buf=self.forward_metadata.decode_wrappers[1]._paged_kv_indices_buf,
                    num_kv_heads=self.num_kv_heads,
                    block_size=self.block_size,
                    num_blocks_per_page=self.num_blocks_per_page,
                    policy_str=_policy,
                )

            # Sparse attention compute
            if os.environ.get("VORTEX_USE_CUSTOM") == "1":
                # Option A: custom Triton kernel — gather + flash-attention
                # fused, no FlashInfer wrapper. Reads the same sparse_kv_indptr
                # / sparse_kv_indices that the indexer just filled.
                # q.shape = [bs * num_kv_heads, group_size, head_dim]
                bsr_M = q.shape[0]
                total = int(self.ctx.metadata.sparse_kv_indptr[bsr_M].item())
                # cache["k"] / cache["v"] viewed as [num_blocks, block_size, head_dim]
                ck = cache["k"].view(-1, self.block_size, self.head_dim)
                cv = cache["v"].view(-1, self.block_size, self.head_dim)
                o = self.custom_decode.run(
                    q=q,
                    cache_k=ck,
                    cache_v=cv,
                    indptr=self.ctx.metadata.sparse_kv_indptr[:bsr_M + 1],
                    indices=self.forward_metadata.decode_wrappers[1]._paged_kv_indices_buf[:total],
                    last_block_len=self.ctx.metadata.kv_last_page_len[:bsr_M],
                    sm_scale=layer.scaling,
                )
                if os.environ.get("VORTEX_CUSTOM_DEBUG") == "1" and layer.layer_id == 1:
                    import sys
                    print(f"[CUSTOM layer={layer.layer_id}] q.shape={tuple(q.shape)} cache_k.shape={tuple(cache['k'].shape)} bsr_M={bsr_M} total_indices={total}", file=sys.stderr)
                    print(f"[CUSTOM layer={layer.layer_id}] o stats: mean={o.float().mean().item():.4f} std={o.float().std().item():.4f} "
                          f"min={o.float().min().item():.4f} max={o.float().max().item():.4f} nans={torch.isnan(o).any().item()}", file=sys.stderr)
            elif os.environ.get("VORTEX_USE_BSR") == "1":
                # Option C: use BSR wrapper with flat KV view (no page concept).
                # Must plan BSR PER LAYER (after indexer fills indices), because
                # BlockSparseAttentionWrapper snapshots indices values at plan
                # time and has no buffer-pointer reuse mode.
                self.bsr_wrapper.plan(
                    indptr=self.ctx.metadata.sparse_kv_indptr[:self._bsr_M+1],
                    indices=self.forward_metadata.decode_wrappers[1]._paged_kv_indices_buf[:int(self.ctx.metadata.sparse_kv_indptr[self._bsr_M].item())],
                    M=self._bsr_M,
                    N=self._bsr_N,
                    R=1,
                    C=self.block_size,
                    num_qo_heads=self.group_size,
                    num_kv_heads=1,
                    head_dim=self.head_dim,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type,
                    o_data_type=self.q_data_type,
                )
                # cache["k"] shape: [total_blocks, block_size, head_dim] -> flatten
                # to [total_blocks * block_size, num_kv_heads=1, head_dim] for BSR.
                k_flat = cache["k"].view(-1, 1, self.head_dim)
                v_flat = cache["v"].view(-1, 1, self.head_dim)
                o = self.bsr_wrapper.run(q, k_flat, v_flat)
                # DEBUG: log first call's shapes + output stats
                if os.environ.get("VORTEX_BSR_DEBUG") == "1" and layer.layer_id == 1:
                    import sys
                    print(f"[BSR layer={layer.layer_id}] q.shape={tuple(q.shape)} q.dtype={q.dtype}", file=sys.stderr)
                    print(f"[BSR layer={layer.layer_id}] k_flat.shape={tuple(k_flat.shape)} k_flat.dtype={k_flat.dtype}", file=sys.stderr)
                    print(f"[BSR layer={layer.layer_id}] o.shape={tuple(o.shape)} o.dtype={o.dtype}", file=sys.stderr)
                    print(f"[BSR layer={layer.layer_id}] o stats: mean={o.float().mean().item():.4f} std={o.float().std().item():.4f} "
                          f"min={o.float().min().item():.4f} max={o.float().max().item():.4f} nans={torch.isnan(o).any().item()}", file=sys.stderr)
            else:
                o = self.forward_metadata.decode_wrappers[1].forward(
                    q,
                    (cache_k, cache_v),
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                )

        else:
            # Dense attention path
            o = self.forward_metadata.decode_wrappers[0].forward(
                q.contiguous().view(-1, self.group_size, layer.head_dim),
                (cache_k, cache_v),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
            )

        # Restore to merged head dimension
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)