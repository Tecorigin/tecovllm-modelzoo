"""Attention layer with FlashAttention."""

import math
import copy
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)

from vllm.config import (
    VllmConfig,
)
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_sdaa.logger import init_logger

logger = init_logger(__name__)

def reshape_and_cache_flash(
    key, # [num_tokens, num_kv_heads, head_size]
    value,
    key_cache, # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache,
    slot_mapping,
):
    num_blocks, block_size, num_kv_heads, head_size = key_cache.size()
    key_cache = key_cache.view(num_blocks * block_size, num_kv_heads, head_size)
    value_cache = value_cache.view(num_blocks * block_size, num_kv_heads, head_size)
    for i in range(slot_mapping.numel()):
        slot_idx = int(slot_mapping[i])
        if slot_idx == -1:
            continue
        else:
            key_cache[slot_idx] = key[i]
            value_cache[slot_idx] = value[i]

def flash_attn_varlen_func(
    q, # [num_tokens, num_heads, head_size]
    k, # [num_blocks, block_size, num_kv_heads, head_size]
    v,
    max_seqlen_q,
    cu_seqlens_q, # [batch_size + 1]
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None, # [batch_size]
    softmax_scale=None,
    causal=False,
    window_size: list[int] | None = None,
    block_table=None, # [batch_size, block_table_dim]
    return_softmax_lse=False,
    out=None,
):
    assert cu_seqlens_k is None
    assert seqused_k is not None
    assert causal
    assert block_table is not None
    assert not return_softmax_lse

    def cdiv(x, y):
        return (x + y - 1) // y

    softmax_scale = 1 / math.sqrt(q.size(-1)) if softmax_scale is None else softmax_scale

    _, num_heads, _ = q.size()
    num_blocks, block_size, num_kv_heads, head_size = k.size()

    for i in range(cu_seqlens_q.numel() - 1):
        L = int(cu_seqlens_q[i+1]) - int(cu_seqlens_q[i])
        S = int(seqused_k[i])
        block_ids = block_table[i, :cdiv(S, block_size)]

        out_ = out[int(cu_seqlens_q[i]):int(cu_seqlens_q[i+1])]
        q_ = q[int(cu_seqlens_q[i]):int(cu_seqlens_q[i+1])]
        k_ = k.index_select(0, block_ids).view(-1, num_kv_heads, head_size)[:S]
        v_ = v.index_select(0, block_ids).view(-1, num_kv_heads, head_size)[:S]

        attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
        if causal:
            attn_mask = torch.ones(S, S, dtype=torch.bool, device=q.device).tril(diagonal=0).logical_not()[-L:]
            attn_bias = attn_bias.masked_fill_(attn_mask, float("-inf"))

        q_ = q_.permute(1, 0, 2)
        k_ = k_.permute(1, 2, 0).repeat_interleave(num_heads // num_kv_heads, 0)
        p = torch.bmm(q_, k_) * softmax_scale + attn_bias
        p = torch.softmax(p, dim=-1)
        v_ = v_.permute(1, 0, 2).repeat_interleave(num_heads // num_kv_heads, 0)
        o = torch.bmm(p, v_).permute(1, 0, 2)
        out_.copy_(o, non_blocking=True)

class FlashAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    forward_includes_kv_cache_update: bool = False

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return super().get_preferred_block_size(default_block_size)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
        )

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return True

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in ["auto", "float16"]

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    causal: bool = True

class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=causal,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.attn_type = attn_type
        logger.info_once(
            "Using FlashAttention",
        )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        if sinks is not None:
            raise NotImplementedError(
                "FlashAttention does not support sinks on this device."
            )

        if logits_soft_cap is not None:
            raise NotImplementedError(
                "FlashAttention does not support logits_soft_cap on this device."
            )

        if alibi_slopes is not None:
            raise NotImplementedError(
                "FlashAttention does not support alibi_slopes on this device."
            )

        if sliding_window is not None:
            raise NotImplementedError(
                "FlashAttention does not support sliding_window on this device."
            )
        self.sliding_window = sliding_window

        if attn_type is not AttentionType.DECODER:
            raise NotImplementedError(
                "FlashAttention only support AttentionType.DECODER on this device."
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        sliding_window_size = (
            list(self.sliding_window)
            if self.sliding_window is not None
            else None
        )
        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=sliding_window_size,
            block_table=block_table,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        key_cache, value_cache = kv_cache.unbind(0)

        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
        )
