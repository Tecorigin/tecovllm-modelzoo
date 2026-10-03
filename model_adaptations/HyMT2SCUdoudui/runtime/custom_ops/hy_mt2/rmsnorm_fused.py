#!/usr/bin/env python3
"""Hy-MT2-1.8B 融合 RMSNorm overlay（SDAA，启动期注入）。

背景（实测，见 experiments/hy-mt2-1.8b.md）：
- transformers `HunYuanDenseV1RMSNorm.forward` 在 SDAA 上未融合，decode 单次
  hidden RMSNorm 约 36 us / 7 kernel，QK-Norm 约 52 us / 7 kernel。
- 厂商栈 `tecoops.rms_norm`（`libteco_ops.so` / `libtecocustom.so`）提供融合
  RMSNorm，但**只支持 fp16**（fp32/bf16 会产出错误值，实测确认）。
- 模型以 bf16 运行，故本 overlay 在 bf16<->fp16 之间显式转换后调用融合算子：
  `bf16 -> fp16 (cast) -> fused rms_norm (1 kernel) -> fp16 -> bf16 (cast)`。

设计约束：
- 热路径零分支：是否启用、权重 fp16 副本、eps 均在 `apply_rmsnorm_overlay`
  进程初始化时一次性绑定，forward 闭包内无 if/else、无 getenv。
- 不支持的 dtype / 设备保持 transformers 原实现，不静默降级。
- 仅当输入为 2D 可展平时调用；模型热路径张量均连续，`reshape(-1, H)` 为视图。

用法：
    from custom_ops.hy_mt2 import apply_rmsnorm_overlay
    n = apply_rmsnorm_overlay(model)          # model 已在目标 device/dtype 上
"""

import logging

import torch

logger = logging.getLogger(__name__)

_TECO = None
_TECO_TRIED = False


def _tecoops():
    """惰性加载 tecoops；仅在 overlay 绑定时调用一次，不在热路径调用。"""
    global _TECO, _TECO_TRIED
    if not _TECO_TRIED:
        _TECO_TRIED = True
        try:
            import tecoops  # noqa: F401

            _TECO = tecoops
        except Exception as exc:  # pragma: no cover - 环境相关
            logger.warning("tecoops unavailable, RMSNorm overlay disabled: %s", exc)
            _TECO = None
    return _TECO


def fused_rms_norm(hidden_states, weight, eps, weight16=None):
    """参考实现：bf16/fp16 输入 → 厂商融合 RMSNorm。

    该函数用于正确性测试与通用调用，内部含 dtype 分支；**模型热路径使用
    `_make_fused_forward` 绑定出的无分支闭包**。fp32 不支持，直接报错。
    """
    tecoops = _tecoops()
    if tecoops is None:
        raise RuntimeError("tecoops not available")

    orig_dtype = hidden_states.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"fused RMSNorm supports fp16/bf16 only, got {orig_dtype}")

    hidden = hidden_states.shape[-1]
    x2d = hidden_states.reshape(-1, hidden)
    x16 = x2d if orig_dtype == torch.float16 else x2d.to(torch.float16)
    w16 = weight16 if weight16 is not None else weight.to(torch.float16)
    out16 = torch.empty_like(x16)
    tecoops.rms_norm(x16, w16, None, out16, None, float(eps))
    out = out16.view(hidden_states.shape)
    return out if orig_dtype == torch.float16 else out.to(orig_dtype)


def _make_fused_forward(weight16, eps, out_dtype, tecoops):
    """生成无分支热路径闭包：bf16 输入 → cast → 融合 kernel → cast 回。"""

    def forward(hidden_states):
        hidden = hidden_states.shape[-1]
        x16 = hidden_states.reshape(-1, hidden).to(torch.float16)
        out16 = torch.empty_like(x16)
        tecoops.rms_norm(x16, weight16, None, out16, None, eps)
        return out16.view(hidden_states.shape).to(out_dtype)

    return forward


def apply_rmsnorm_overlay(model, verbose=True):
    """把 `HunYuanDenseV1RMSNorm.forward` 替换为融合实现，返回绑定模块数。

    只在权重为 bf16 且位于 sdaa 设备上时绑定；其它情况保持原实现。
    """
    tecoops = _tecoops()
    if tecoops is None:
        if verbose:
            print("[overlay] tecoops not available; RMSNorm overlay skipped")
        return 0

    from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
        HunYuanDenseV1RMSNorm,
    )

    bound = 0
    skipped = 0
    for module in model.modules():
        if not isinstance(module, HunYuanDenseV1RMSNorm):
            continue
        weight = module.weight
        if weight.dtype != torch.bfloat16 or weight.device.type != "sdaa":
            skipped += 1
            continue
        # 一次性绑定：fp16 权重副本 + eps + 输出 dtype + tecoops 引用
        weight16 = weight.detach().to(torch.float16).contiguous()
        module.forward = _make_fused_forward(
            weight16, float(module.variance_epsilon), torch.bfloat16, tecoops
        )
        module._hy_fused_rmsnorm = True
        bound += 1

    if verbose:
        print(
            f"[overlay] fused RMSNorm bound on {bound} modules "
            f"(skipped {skipped} non-bf16/sdaa)"
        )
    return bound


def remove_rmsnorm_overlay(model):
    """移除实例级 forward 覆盖，恢复类方法（用于对照测试）。"""
    from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
        HunYuanDenseV1RMSNorm,
    )

    removed = 0
    for module in model.modules():
        if isinstance(module, HunYuanDenseV1RMSNorm) and "_hy_fused_rmsnorm" in module.__dict__:
            del module.__dict__["_hy_fused_rmsnorm"]
            module.__dict__.pop("forward", None)
            removed += 1
    return removed
