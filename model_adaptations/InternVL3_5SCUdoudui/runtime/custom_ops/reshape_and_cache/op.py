"""custom_ops/reshape_and_cache/op.py - 官方 reshape_and_cache 算子封装与基准

依据：
- main.pdf Section 5.1, 9.2
- 实施后审查修复任务书（2026-09-20）任务 5 与修复 7.1
- MiniCPM5-1B 规格：num_heads=2 (GQA), head_size=128, block_size=32, dtype=float16
"""

import time
from typing import Any, Optional, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import torch_sdaa
    import tecoops
    HAS_TECO_OPS = True
except Exception:
    HAS_TECO_OPS = False

from custom_ops.receipt import bind_official_api


def _tecoops_reshape_and_cache(
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
) -> None:
    """Call the official tecoops ABI bound at import time.

    调用计数由 ``custom_ops.receipt`` 在导入期一次性绑定：未设置
    ``TECOOPS_CALL_RECEIPT`` 时这里就是官方函数本体，热路径没有计数器、分支或
    ``getenv``；设置时按 ``tecoops.reshape_and_cache`` 这个名字累计调用数
    （用于把 profiler 的 720 次与运行时真实调用对上）。
    """
    # The official C++ ABI reads key/value as dense half buffers and does not
    # consume PyTorch stride metadata.  vLLM fused-QKV views (especially
    # ``value``) are commonly non-contiguous, so materialize only at this ABI
    # boundary before the kernel performs its own linear address arithmetic.
    tecoops.reshape_and_cache(
        key.contiguous(),
        value.contiguous(),
        slot_mapping,
        key_cache,
        value_cache,
    )


def _infer_block_size(key_cache: Any, value_cache: Any, explicit: Optional[int] = None) -> int:
    """推断 KV cache 的 ``block_size``。

    **这里此前是错的**：旧代码写 ``value_cache.shape[-1] if value_cache.ndim == 4
    else value_cache.shape[2]``，在本项目实测的 **HND** 排布下
    （key/value 均为 ``[num_blocks, num_heads, block_size, head_size]``）
    会把 ``head_size = 128`` 当成 ``block_size``，于是槽位地址整体算错。
    参考实现只用于离线单测，所以这个错一直没被暴露。

    实测口径（`vllm_sdaa/attention/block_attn.py:41-85` 为准）：

    * 5D SDAA 硬件排布（仅 key_cache）：``[nb, nh, head_size//8, block_size, 8]``
      → ``block_size = shape[3]``
    * 4D HND（本项目实际使用）：``[nb, nh, block_size, head_size]`` → ``shape[2]``
    """
    if explicit is not None:
        return int(explicit)
    if key_cache.ndim == 5:
        return int(key_cache.shape[3])
    if key_cache.ndim == 4:
        return int(key_cache.shape[2])
    if value_cache.ndim == 4:
        return int(value_cache.shape[3])
    raise ValueError(
        "cannot infer block_size: key_cache.ndim="
        f"{key_cache.ndim}, value_cache.ndim={value_cache.ndim}"
    )


def _reference_reshape_and_cache(
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
    block_size: Optional[int] = None,
) -> None:

    """太初 SDAA reshape_and_cache 参考实现（**仅离线单测用**）

    参数:
        key: [num_tokens, num_heads, head_size], float16
        value: [num_tokens, num_heads, head_size], float16
        key_cache: [num_blocks, num_heads, head_size // 8, block_size, 8], float16 (SDAA 硬件排布)
                   或 [num_blocks, num_heads, block_size, head_size]（HND，本项目实际使用）
        value_cache: [num_blocks, num_heads, head_size, block_size]
                     或 [num_blocks, num_heads, block_size, head_size]（HND）
        slot_mapping: [num_tokens], int64 槽位映射；**负值表示该 token 不写缓存**
        block_size: 可选显式覆盖；不给则按上面的排布推断
    """
    # 参考 CPU/CUDA 实现，仅用于离线单元测试与逻辑验证。
    block_size = _infer_block_size(key_cache, value_cache, block_size)
    num_tokens, num_heads, head_size = key.shape
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        if slot < 0:
            continue
        b = slot // block_size
        offset = slot % block_size
        if key_cache.ndim == 5:
            for h in range(num_heads):
                for d in range(head_size):
                    key_cache[b, h, d // 8, offset, d % 8] = key[i, h, d]
        elif key_cache.ndim == 4:
            key_cache[b, :, offset, :] = key[i]

        if value_cache.ndim == 4 and value_cache.shape[-1] == block_size:
            for h in range(num_heads):
                for d in range(head_size):
                    value_cache[b, h, d, offset] = value[i, h, d]
        elif value_cache.ndim == 4:
            value_cache[b, :, offset, :] = value[i]


sdaa_reshape_and_cache = (
    bind_official_api(_tecoops_reshape_and_cache, "tecoops.reshape_and_cache")
    if HAS_TECO_OPS
    else _reference_reshape_and_cache
)


def bench_reshape_and_cache(
    num_tokens: int = 128,
    num_heads: int = 2,
    head_size: int = 128,
    num_blocks: int = 64,
    block_size: int = 32,
    device: str = "sdaa:0",
    trials: int = 3,
    warmup: int = 5,
) -> Tuple[list, float]:
    """对 MiniCPM5-1B 真实 shape 执行同口径 3 次基准测量并返回中位数"""
    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
    val = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device=device)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)

    if HAS_TECO_OPS:
        key_cache = torch.zeros(
            num_blocks,
            num_heads,
            block_size,
            head_size,
            dtype=torch.float16,
            device=device,
        )
        val_cache = torch.zeros_like(key_cache)
    else:
        key_cache = torch.zeros(
            num_blocks,
            num_heads,
            head_size // 8,
            block_size,
            8,
            dtype=torch.float16,
            device=device,
        )
        val_cache = torch.zeros(
            num_blocks,
            num_heads,
            head_size,
            block_size,
            dtype=torch.float16,
            device=device,
        )

    # 预热
    for _ in range(warmup):
        sdaa_reshape_and_cache(key, val, key_cache, val_cache, slots)
    if "sdaa" in device and hasattr(torch, "sdaa"):
        torch.sdaa.synchronize()

    times_ms = []
    for _ in range(trials):
        t0 = time.perf_counter()
        sdaa_reshape_and_cache(key, val, key_cache, val_cache, slots)
        if "sdaa" in device and hasattr(torch, "sdaa"):
            torch.sdaa.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(round(elapsed_ms, 4))

    median_ms = sorted(times_ms)[len(times_ms) // 2]
    return times_ms, median_ms
