"""custom_ops/prefill_attention/op.py - 注意力：paged cache gather + torch SDPA

## 为什么是「gather 补齐 + 逐序列掩码」而不是「加一个回退分支」

旧版只吃**当前步的 dense K/V**，因此要求 `query_len == seq_len`；一旦批里混有 decode
序列（`query_len=1`、`seq_len=1812`），或 prompt 被 chunked prefill 切分，就 fail-closed。
那是 CP3 优化引入的回归（见 experiments/internvl3_5-8b.md §13/§14）。

修法**不是**在 forward 里加「有前缀就走另一条实现」的 if-else，而是：
每条序列按自己的 `block_table` + `seq_lens` 从 paged cache **gather 出完整 K/V**，
再交给**同一个** SDPA。实现只有一条，在 import 期绑定；无后端/fallback/A-B 选择、无 getenv。

## 09-24 实测：SDAA 融合 SDPA 只对「方阵 + causal」快

| shape | is_causal | 耗时 | 派发 |
| :--- | :--- | ---: | :--- |
| 1811×1811 | True | **4.571 ms** | FUSED |
| 1811×1811 | False | 7.667 ms | FUSED |
| 1×1812 | True | 1.456 ms | **MATH（慢）** |
| 1×1812 | **False** | **0.984 ms** | **FUSED** |
| 961×3009 | True | 68.737 ms | **MATH** |
| 961×3009 | False | 5.835 ms | FUSED（但语义错） |

两条结论：
1. **causal + 非方阵 ⇒ math 慢路径** —— 决定因素是**形状**，不是「掩码」。
   所以 `is_causal=True` 在 `q_len < kv_len` 时既慢、又是**左上角对齐（数值错）**。
2. **decode 序列（q_len=1）用 `is_causal=False` 正确且融合** ——
   q=1 时所有 key 都在过去，本就不需要掩码，而且比 `is_causal=True` 更快。
   这正是修掉并发回归的关键。

因此每条序列的掩码取 `causal and (q_len > 1)`：
- 完整 prefill（q_len == kv_len > 1）→ True，方阵，融合；
- 混合批里的 decode（q_len == 1）→ False，非方阵但**非因果**，融合且正确。

## 仍不支持：chunked prefill（1 < q_len < kv_len）

该形状要「右下角因果」，而 SDAA 上三条路都不可用：
`is_causal=True` ⇒ math + 左上角（又慢又错）；显式 `attn_mask` ⇒ 同样 math（68.7 ms 量级）；
`is_causal=False` ⇒ 融合但会看到未来 key（语义错）。
⇒ **显式 fail-closed**，并提示用 `--no-enable-chunked-prefill`
（该旗标下每个 prefill 原子完成，恒有 `q_len == kv_len`，本条件不可达）。

## 布局依据（厂商代码，权威来源）

`vllm_sdaa/attention/block_attn.py:41-85` 与 `BlockAttentionBackend.get_kv_cache_shape`：
SDAA 用 **HND** 布局 `key_cache: [num_blocks, num_kv_heads, block_size, head_size]`，
`block_size = key_cache.size(2)`。本模块自行实现 gather，不绑定厂商内部私有符号。

## ⚠️ 先决条件：`torch.backends.sdaa.enable_flash_sdp(True)`

不开时 `torch_sdaa` **静默**落到 `aten::_scaled_dot_product_attention_math`。
vLLM 在 `vllm_sdaa/worker/sdaa_model_runner.py:143` 会打开；本模块在 **import 时一次性绑定**，
使算子不依赖调用方环境（见 tests test_7）。
"""

import torch
import torch.nn.functional as F

# 进程初始化层一次性绑定：打开 SDAA flash SDP 后端。
# 不开则 torch_sdaa 静默回退到 _scaled_dot_product_attention_math。
# 只在这里绑定一次，forward 热路径不做任何查询或分支。
try:
    import torch_sdaa  # noqa: F401
    torch.backends.sdaa.enable_flash_sdp(True)
except Exception:  # 非 SDAA 环境（例如纯 CPU 单测）不阻断导入
    pass


def gather_kv_from_cache(key_cache, value_cache, block_table_seq, kv_len, block_size):
    """按序列的 block_table 从 paged cache 取出前 kv_len 个 K/V。

    布局：key_cache: [num_blocks, num_kv_heads, block_size, head_size]（SDAA HND）。
    block_table_seq 中超出 kv_len 的槽位（常见是 -1 填充）不会被索引到。
    """
    logical = torch.arange(kv_len, device=key_cache.device)
    block_ids = block_table_seq[logical // block_size]
    offsets = logical % block_size
    return (key_cache[block_ids, :, offsets, :],
            value_cache[block_ids, :, offsets, :])


def sdpa_prefill_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """paged cache gather + torch SDPA（纯 prefill / 混合批 decode 统一；chunked 显式拒绝）。

    参数:
        query:           [num_tokens, num_heads_q,  head_size]
        key_cache:       [num_blocks, num_heads_kv, block_size, head_size]
        value_cache:     [num_blocks, num_heads_kv, block_size, head_size]
        block_table:     [num_seqs, max_blocks] int32
        query_start_loc: [num_seqs + 1] int32，Q 的累积长度
        seq_lens:        [num_seqs] int32，每序列 K/V 总长度（含本步与前缀）
        scale:           softmax 缩放
        causal:          模型是否因果

    返回:
        out: [num_tokens, num_heads_q, head_size]
    """
    out = torch.empty_like(query)
    num_seqs = query_start_loc.numel() - 1
    block_size = key_cache.size(2)
    # GQA 开关按 shape 一次性判定，不随序列变化
    enable_gqa = query.shape[1] != key_cache.shape[1]

    for i in range(num_seqs):
        q_start = int(query_start_loc[i])
        q_end = int(query_start_loc[i + 1])
        q_len = q_end - q_start
        kv_len = int(seq_lens[i])
        if kv_len <= 0 or q_len <= 0:
            continue
        # chunked prefill：需要右下角因果，SDAA 上三条路皆不可用（见模块头）
        if causal and q_len > 1 and q_len != kv_len:
            raise RuntimeError(
                "custom_ops.prefill_attention: 不支持 chunked prefill "
                f"(sequence {i}: query_len={q_len} != seq_len={kv_len})。"
                "SDAA 上 causal+非方阵只能走 math 慢路径且是左上角对齐（数值错误）。"
                "请用 --no-enable-chunked-prefill 使每个 prefill 原子完成。"
            )
        k_i, v_i = gather_kv_from_cache(
            key_cache, value_cache, block_table[i], kv_len, block_size)
        # q_len == 1 时全部 key 都在过去，无需掩码；且此时非因果才是融合路径
        seq_causal = bool(causal and q_len > 1)
        q_i = query[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_i = k_i.transpose(0, 1).unsqueeze(0)
        v_i = v_i.transpose(0, 1).unsqueeze(0)
        o_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i, is_causal=seq_causal, scale=scale, enable_gqa=enable_gqa)
        out[q_start:q_end] = o_i.squeeze(0).transpose(0, 1)

    return out
