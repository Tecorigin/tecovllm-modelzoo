"""custom_ops/rms_norm/op.py - 官方 RMSNorm / Fused-Add RMSNorm 算子封装与基准

依据：
- main.pdf Section 5.1, 9.2
- 实施后审查修复任务书（2026-09-20）任务 5 与修复 7.2
- MiniCPM5-1B 规格：hidden_size=1536, eps=1e-6, dtype=float16
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
    HAS_TECOOPS = hasattr(tecoops, "rms_norm")
except Exception:
    HAS_TECOOPS = False

_RECEIPT_PATH = os.environ.get("TECOOPS_CALL_RECEIPT")
_RECORDED_CALLS: set[str] = set()


def _tecoops_rms_norm(
    x: Any,
    weight: Any,
    eps: float = 1e-6,
    residual: Optional[Any] = None,
) -> Union[Any, Tuple[Any, Any]]:
    """Call the official tecoops ABI bound at import time."""
    if _RECEIPT_PATH and "tecoops.rms_norm" not in _RECORDED_CALLS:
        _RECORDED_CALLS.add("tecoops.rms_norm")
        with open(_RECEIPT_PATH, "a", encoding="utf-8") as receipt:
            receipt.write("CALL tecoops.rms_norm\n")
    original_shape = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    residual2 = residual.reshape_as(x2) if residual is not None else None
    out2 = torch.empty_like(x2)
    residual_out2 = torch.empty_like(x2) if residual2 is not None else None
    tecoops.rms_norm(x2, weight, residual2, out2, residual_out2, float(eps))
    out = out2.reshape(original_shape)
    if residual_out2 is not None:
        return out, residual_out2.reshape(original_shape)
    return out


def _reference_rms_norm(
    x: Any,
    weight: Any,
    eps: float = 1e-6,
    residual: Optional[Any] = None,
) -> Union[Any, Tuple[Any, Any]]:
    """离线单测使用的 RMSNorm / Fused Add 参考实现。

    参数:
        x: [..., hidden_size], float16
        weight: [hidden_size], float16
        eps: 归一化平滑常数 (默认 1e-6)
        residual: 可选残差项 [..., hidden_size], 若提供则返回 (out, new_residual)

    返回:
        若 residual 为 None: 返回归一化后张量 out
        若 residual 不为 None: 返回 (out, residual_out) 元组
    """
    if residual is not None:
        x = x + residual
        residual_out = x

    if HAS_TORCH and x.is_sdaa and hasattr(torch.ops.aten, "_fused_rms_norm"):
        normalized_shape = [x.shape[-1]]
        out, _ = torch.ops.aten._fused_rms_norm(x, normalized_shape, weight, eps)
    else:
        # 参考 CPU/CUDA 高精度实现
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        out = (x_fp32 * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)

    if residual is not None:
        return out, residual_out
    return out


sdaa_rms_norm = _tecoops_rms_norm if HAS_TECOOPS else _reference_rms_norm


def bench_rms_norm(
    batch_size: int = 4,
    seq_len: int = 128,
    hidden_size: int = 1536,
    eps: float = 1e-6,
    with_residual: bool = True,
    device: str = "sdaa:0",
    trials: int = 3,
    warmup: int = 10,
) -> Tuple[list, float]:
    """对 MiniCPM5-1B 真实 shape (hidden_size=1536) 执行同口径 3 次基准测量并返回中位数"""
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device)
    w = torch.randn(hidden_size, dtype=torch.float16, device=device)
    res = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=device) if with_residual else None

    # 预热
    for _ in range(warmup):
        sdaa_rms_norm(x, w, eps, residual=res)
    if "sdaa" in device and hasattr(torch, "sdaa"):
        torch.sdaa.synchronize()

    times_ms = []
    for _ in range(trials):
        t0 = time.perf_counter()
        sdaa_rms_norm(x, w, eps, residual=res)
        if "sdaa" in device and hasattr(torch, "sdaa"):
            torch.sdaa.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(round(elapsed_ms, 4))

    median_ms = sorted(times_ms)[len(times_ms) // 2]
    return times_ms, median_ms
