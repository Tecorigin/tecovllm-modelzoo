"""SDAA BlockAttention operator framework adaptation.

Provides in-repo implementation for BlockAttentionImpl.forward to bridge
vLLM V1 attention execution to vendor C++ kernels:
- torch.ops._C_cache_ops.reshape_and_cache_flash
- torch.ops._C_sdaa.block_attention
"""

from typing import Optional
import torch

from custom_ops.flash_attn_varlen.op import sdaa_flash_attn_varlen_func
from custom_ops.reshape_and_cache.op import sdaa_reshape_and_cache


def sdaa_block_attention_forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    output_block_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute block attention forward on SDAA accelerator.

    Zero runtime branching in hot path: bindings are established at process
    initialization, no fallback or getenv calls per forward iteration.
    """
    if attn_metadata is None:
        # Profiling / dummy run
        if output is not None:
            return output.fill_(0)
        return torch.zeros(
            (query.shape[0], self.num_heads * self.head_size),
            dtype=query.dtype,
            device=query.device,
        )

    num_actual_tokens = attn_metadata.num_actual_tokens
    if num_actual_tokens == 0:
        return output

    key_cache, value_cache = kv_cache[0], kv_cache[1]
    num_blocks_stride = key_cache.stride(0)

    # Cache new key/value entries into paged KV blocks
    if key is not None and value is not None and attn_metadata.slot_mapping is not None:
        sdaa_reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
        )

    if output is None:
        output = torch.empty(
            (num_actual_tokens, self.num_heads, self.head_size),
            dtype=query.dtype,
            device=query.device,
        )

    out_view = output[:num_actual_tokens]
    if out_view.ndim == 2:
        out_view = out_view.view(-1, self.num_heads, self.head_size)

    if attn_metadata.max_query_len > 1:
        # Prefill uses the official teco-ops paged-cache flash ABI.
        sdaa_flash_attn_varlen_func(
            query[:num_actual_tokens].contiguous(),
            key_cache,
            value_cache,
            attn_metadata.query_start_loc,
            attn_metadata.query_start_loc,
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            seqused_k=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table,
            out=out_view,
        )
    else:
        # Decode is a single-token path; retain the vendor BlockAttention
        # kernel because the official flash ABI is a prefill contract and is
        # not an efficient decode implementation.
        torch.ops._C_sdaa.block_attention(
            out_view,
            key_cache,
            value_cache,
            query[:num_actual_tokens].contiguous(),
            attn_metadata.seq_lens_pre_cache,
            attn_metadata.seq_lens_pre_cache_cpu,
            attn_metadata.seq_lens_prefill,
            attn_metadata.seq_lens_prefill_cpu,
            attn_metadata.seq_lens_decode,
            attn_metadata.query_start_loc,
            attn_metadata.block_table,
            attn_metadata.max_model_len,
            attn_metadata.max_prefill_len,
            attn_metadata.max_decode_len,
            attn_metadata.version,
            window_size_left=self.sliding_window[0],
            window_size_right=self.sliding_window[1],
            sinks=self.sinks,
            block_stride=num_blocks_stride,
            is_q_only=True,
        )

    return output
