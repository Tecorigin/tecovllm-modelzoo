"""custom_ops/flash_attn_varlen/op.py - 官方 flash_attn_varlen_func 算子封装与基准

依据：
- main.pdf Section 5.1, 9.2
- 实施后审查修复任务书（2026-09-20）任务 5 与修复 7.3
- MiniCPM5-1B prefill 变长规格：
  num_heads_q=16, num_heads_kv=2, head_size=128, dtype=float16
"""

import os
import time
from typing import Any, Optional, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import torch_sdaa
    HAS_SDAA = True
except Exception:
    HAS_SDAA = False

try:
    import tecoops
    HAS_TECOOPS = hasattr(tecoops, "flash_attn_varlen_func")
except Exception:
    HAS_TECOOPS = False

_RECEIPT_PATH = os.environ.get("TECOOPS_CALL_RECEIPT")
_RECORDED_CALLS: set[str] = set()


def _record_call(name: str) -> None:
    """Write one process-init diagnostic receipt per official API."""
    if _RECEIPT_PATH and name not in _RECORDED_CALLS:
        _RECORDED_CALLS.add(name)
        with open(_RECEIPT_PATH, "a", encoding="utf-8") as receipt:
            receipt.write(f"CALL {name}\n")


def _tecoops_flash_attn_varlen_func(
    q: Any,
    k: Any,
    v: Any,
    cu_seqlens_q: Any,
    cu_seqlens_k: Any,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    seqused_k: Any = None,
    window_size: Any = None,
    block_table: Any = None,
    out: Any = None,
) -> Any:
    """Call the official paged-cache tecoops ABI bound at import time."""
    _record_call("tecoops.flash_attn_varlen_func")
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    if out is None:
        out = torch.empty_like(q)
    empty = torch.Tensor()
    tecoops.flash_attn_varlen_func(
        q,
        k,
        v,
        int(max_seqlen_q),
        cu_seqlens_q,
        int(max_seqlen_k),
        empty,
        seqused_k,
        float(softmax_scale),
        bool(causal),
        empty,
        block_table,
        False,
        out,
    )
    return out


def _reference_flash_attn_varlen_func(
    q: Any,
    k: Any,
    v: Any,
    cu_seqlens_q: Any,
    cu_seqlens_k: Any,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    seqused_k: Any = None,
    window_size: Any = None,
    block_table: Any = None,
    out: Any = None,
) -> Any:
    """离线单测使用的 packed-tensor 参考实现。

    参数:
        q: [total_tokens_q, num_heads_q, head_size], float16
        k: [total_tokens_k, num_heads_kv, head_size], float16
        v: [total_tokens_k, num_heads_kv, head_size], float16
        cu_seqlens_q: [batch_size + 1], int32
        cu_seqlens_k: [batch_size + 1], int32
        max_seqlen_q: q 的最大序列长度
        max_seqlen_k: k 的最大序列长度
        softmax_scale: 缩放因子，默认 1.0 / sqrt(head_size)
        causal: 是否施加因果掩码 (默认 True)

    返回:
        out: [total_tokens_q, num_heads_q, head_size], float16
    """
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    head_size = q.shape[2]

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_size ** 0.5)

    gqa_ratio = num_heads_q // num_heads_kv
    batch_size = len(cu_seqlens_q) - 1
    out = torch.empty_like(q)

    # 针对变长序列在硬件加速设备上的序列分段处理
    for b in range(batch_size):
        sq, eq = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        sk, ek = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
        if eq <= sq or ek <= sk:
            continue

        qb = q[sq:eq].unsqueeze(0).transpose(1, 2)  # [1, Hq, Lq, D]
        kb = k[sk:ek].unsqueeze(0).transpose(1, 2)  # [1, Hkv, Lk, D]
        vb = v[sk:ek].unsqueeze(0).transpose(1, 2)  # [1, Hkv, Lk, D]

        if gqa_ratio > 1:
            kb = kb.repeat_interleave(gqa_ratio, dim=1)
            vb = vb.repeat_interleave(gqa_ratio, dim=1)

        out_b = torch.nn.functional.scaled_dot_product_attention(
            qb, kb, vb, is_causal=causal, scale=softmax_scale
        )
        out[sq:eq] = out_b.transpose(1, 2).squeeze(0)

    return out


# Bind the backend once during process initialization.  Production SDAA
# processes fail closed in sitecustomize before this module is imported, while
# local CPU tests bind the reference implementation without a per-forward
# backend/fallback branch.
sdaa_flash_attn_varlen_func = (
    _tecoops_flash_attn_varlen_func
    if HAS_TECOOPS
    else _reference_flash_attn_varlen_func
)


if HAS_TECOOPS and hasattr(torch, "compiler"):
    torch.compiler.allow_in_graph(sdaa_flash_attn_varlen_func)


def bench_flash_attn_varlen(
    seq_lens: Tuple[int, ...] = (64, 128),
    num_heads_q: int = 16,
    num_heads_kv: int = 2,
    head_size: int = 128,
    device: str = "sdaa:0",
    trials: int = 3,
    warmup: int = 10,
) -> Tuple[list, float]:
    """对 MiniCPM5-1B 真实 prefill 变长 shape 执行同口径 3 次基准测量并返回中位数"""
    total_tokens = sum(seq_lens)
    q = torch.randn(total_tokens, num_heads_q, head_size, dtype=torch.float16, device=device)

    cu_list = [0]
    for length in seq_lens:
        cu_list.append(cu_list[-1] + length)
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=device)
    max_len = max(seq_lens)

    if HAS_TECOOPS:
        block_size = 32
        blocks_per_sequence = [
            (length + block_size - 1) // block_size for length in seq_lens
        ]
        num_blocks = sum(blocks_per_sequence)
        max_blocks = max(blocks_per_sequence)
        k = torch.randn(
            num_blocks,
            num_heads_kv,
            block_size,
            head_size,
            dtype=torch.float16,
            device=device,
        )
        v = torch.randn_like(k)
        block_table = torch.full(
            (len(seq_lens), max_blocks),
            -1,
            dtype=torch.int32,
            device=device,
        )
        first_block = 0
        for batch_index, block_count in enumerate(blocks_per_sequence):
            block_table[batch_index, :block_count] = torch.arange(
                first_block,
                first_block + block_count,
                dtype=torch.int32,
                device=device,
            )
            first_block += block_count
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        call_args = {
            "seqused_k": seq_lens_tensor,
            "block_table": block_table,
        }
    else:
        k = torch.randn(total_tokens, num_heads_kv, head_size, dtype=torch.float16, device=device)
        v = torch.randn(total_tokens, num_heads_kv, head_size, dtype=torch.float16, device=device)
        call_args = {}

    # 预热
    for _ in range(warmup):
        sdaa_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_len,
            max_len,
            causal=True,
            **call_args,
        )
    if "sdaa" in device and hasattr(torch, "sdaa"):
        torch.sdaa.synchronize()

    times_ms = []
    for _ in range(trials):
        t0 = time.perf_counter()
        sdaa_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_len,
            max_len,
            causal=True,
            **call_args,
        )
        if "sdaa" in device and hasattr(torch, "sdaa"):
            torch.sdaa.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(round(elapsed_ms, 4))

    median_ms = sorted(times_ms)[len(times_ms) // 2]
    return times_ms, median_ms
