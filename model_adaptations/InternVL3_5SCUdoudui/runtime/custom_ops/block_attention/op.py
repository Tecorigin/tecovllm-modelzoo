"""SDAA BlockAttention operator framework adaptation.

Provides in-repo implementation for BlockAttentionImpl.forward to bridge
vLLM V1 attention execution to vendor C++ kernels:
- torch.ops._C_cache_ops.reshape_and_cache_flash
- torch.ops._C_sdaa.block_attention
"""

from typing import Optional
import torch

from custom_ops.prefill_attention.op import sdpa_prefill_attention
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
        # Prefill / 混合批：按 block_table 从 paged cache gather 完整 K/V，再交给 SDPA。
        # 见 custom_ops/prefill_attention/op.py。
        # 09-24 kernel 级取证：tecoops 的 paged flash ABI 在本模型真实 prefill shape
        # (seq=1811, hq=16, hkv=4, d=128) 上只有 ~60 GFLOPS，且耗时对 block_size
        # (16/32/64)、max_seqlen_k、dense/paged 布局**不敏感** —— 低效在厂商 paged kernel
        # 内部。同 shape 的 torch SDPA 派发到融合厂商 kernel，实测 224.1 ms -> 4.839 ms/call。
        # 这是**唯一一条**实现（不是运行时 A/B）：没有开关、没有 getenv、没有回退分支。
        out_view.copy_(sdpa_prefill_attention(
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
            self.scale,
            attn_metadata.causal,
        ))
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
