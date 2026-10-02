"""SDAA BlockAttention operator framework adaptation.

Provides in-repo implementation for BlockAttentionImpl.forward to bridge
vLLM V1 attention execution to vendor C++ kernels:
- torch.ops._C_cache_ops.reshape_and_cache_flash
- torch.ops._C_sdaa.block_attention
"""

import functools
from typing import Optional
import torch

from custom_ops.prefill_attention.op import (
    sdpa_prefill_attention,
    sdpa_prefill_attention_math,
    sdpa_decode_attention_math,
)
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
        # Prefill / 混合批：按 block_table 从 paged cache gather 完整 K/V，再交给注意力。
        # 见 custom_ops/prefill_attention/op.py。
        # 实现由 **构造期** 绑定（self._gemma4_prefill_fn）：Gemma 的 D256
        # sliding 与 D512 global 层均走自研 paged-cache math kernel；窗口语义在
        # 构造期绑定，forward 内无后端选择分支。
        out_view.copy_(self._gemma4_prefill_fn(
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
        # Decode 单 token 路径。实现同样在构造期绑定（self._gemma4_decode_fn）：
        #   head_size<=256 -> 厂商 _C_sdaa.block_attention kernel
        #   head_size>256  -> 同一套 gather+math（厂商 kernel/SDP 在 512 上不可用）
        self._gemma4_decode_fn(
            self, out_view, query, key_cache, value_cache,
            attn_metadata, num_actual_tokens, num_blocks_stride,
        )

    return output


def _decode_vendor(self, out_view, query, key_cache, value_cache,
                   attn_metadata, num_actual_tokens, num_blocks_stride):
    """Decode 走厂商 BlockAttention kernel（head_size<=256 时的绑定实现）。"""
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


def _decode_math(self, out_view, query, key_cache, value_cache,
                 attn_metadata, num_actual_tokens, num_blocks_stride):
    """Decode 走窗口裁剪 + 融合 SDPA（D256）或 gather+math（D512）。"""
    out_view.copy_(sdpa_decode_attention_math(
        query[:num_actual_tokens],
        key_cache,
        value_cache,
        attn_metadata.block_table,
        attn_metadata.query_start_loc,
        attn_metadata.seq_lens,
        self.scale,
        window_size=getattr(self, "_gemma4_window_size", None),
    ))


PREFILL_IMPLS = {
    "flash": sdpa_prefill_attention,       # head_size <= 256
    "math": sdpa_prefill_attention_math,   # head_size > 256（厂商 flash 不支持）
}
DECODE_IMPLS = {
    "vendor": _decode_vendor,
    "math": _decode_math,
}
FLASH_MAX_HEAD_DIM = 256

def bind_attention_impls(impl_self) -> None:
    """按 head_size **一次性** 绑定 prefill/decode 实现（进程/构造期调用，非热路径）。

    * Gemma 的 D256 sliding 与 D512 global 层均走显式 math kernel；D256
      的厂商 flash 在真实 GQA shape 上数值错误，D512 则触发厂商 headdim 断言。
    * 窗口大小只在构造期绑定：D256 sliding 保存真实 window=1024，D512 global 为 None。

    为什么 >256 的 decode 不能用厂商 kernel（实测，不是推断）：
      把 decode 一律交给厂商 kernel 后，32-token 短生成的 token_ids 变成
      ``[818, -1, -1, -1, ...]`` —— id 为 -1 表示采样器拿到 NaN/Inf logits。
      厂商 kernel 在 head_dim=512 上**不 assert、但静默产出垃圾**。
    """
    # Gemma's D256 path is also vendor-inaccurate (not merely unsupported):
    # bind the in-repo kernel for both D256 sliding and D512 global layers.
    base_math = impl_self.head_size >= FLASH_MAX_HEAD_DIM
    window_size = getattr(impl_self, "_gemma4_window_size", None)
    if base_math:
        impl_self._gemma4_prefill_fn = functools.partial(
            sdpa_prefill_attention_math, window_size=window_size
        )
    else:
        impl_self._gemma4_prefill_fn = PREFILL_IMPLS["flash"]
    impl_self._gemma4_decode_fn = DECODE_IMPLS["math"] if base_math else DECODE_IMPLS["vendor"]

# ---------------------------------------------------------------------------
# Provenance: ported verbatim from model/internvl3_5-8b
# (commit 2ae15b17f254b7e710c74bb8c64bd45ca2d76218)
# custom_ops/<dir>/op.py, read-only reference. Reason: vllm_sdaa's
# BlockAttentionImpl.forward is a NotImplementedError stub in this vendor build
# (vllm_sdaa/attention/block_attn.py:881), so every SDAA vLLM V1 attention call
# fails at the KV-profiling forward without this overlay.
# ---------------------------------------------------------------------------
