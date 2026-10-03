#!/usr/bin/env python3
"""Hy-MT2-1.8B 融合 SwiGLU MLP overlay（SDAA，启动期注入）。

背景（实测，见 experiments/hy-mt2-1.8b.md 与 step3）：
- transformers `HunYuanDenseV1MLP.forward` = `down_proj(silu(gate_proj(x)) * up_proj(x))`，
  在 SDAA 上分解为 2 个 GEMM + silu + mul（4 次 launch/层）。
- 厂商栈导出融合 SwiGLU：`torch.ops.sdaa.swiglu(self, seq_lens, num_head, hidden_size)`
  （由 `torch_sdaa.nn` 注册，`libtecocustom.so::tecocustomSwigluActivateForward`）。
  实测语义：输入 **4D 连续** `(1, S, B, 2H)`，输出 `(S, B, H)`，
  `out = silu(x[..., :H]) * x[..., H:]`；fp16/bf16/fp32 均支持，
  要求 last dim（`2H`）为 16 的倍数。

设计（单一假设：融合 SwiGLU MLP）：
- 仅把 `silu(g)*u` 换成融合算子（`cat + swiglu`）实测**更慢**——`torch.cat` 的拷贝
  抵消了省下的一次 launch；故采用**合并 gate_proj/up_proj 权重为单个 gate_up GEMM**
  + 融合 swiglu：`gu = x @ [W_gate; W_up]^T -> silu(gu[..., :I]) * gu[..., I:]`。
  microbench（T=1, H=2048, I=6144, bf16）1.709 -> 1.629 ms/call，优于 cat 方案（1.735）。
- 合并权重在 `apply_swiglu_overlay` 绑定期一次性构建；热路径零分支、无 getenv。
- 不改 site-packages / torch / transformers，仅替换 `HunYuanDenseV1MLP` 实例 forward。

用法：
    from custom_ops.hy_mt2 import apply_swiglu_overlay
    n = apply_swiglu_overlay(model)          # model 已在目标 device/dtype 上
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SWIGLU = None
_SWIGLU_TRIED = False


def _sdaa_swiglu():
    """惰性取得融合算子；仅在 overlay 绑定时调用一次，不在热路径调用。"""
    global _SWIGLU, _SWIGLU_TRIED
    if not _SWIGLU_TRIED:
        _SWIGLU_TRIED = True
        try:
            import torch_sdaa  # noqa: F401
            import torch_sdaa.nn  # noqa: F401  注册 torch.ops.sdaa.swiglu

            _SWIGLU = torch.ops.sdaa.swiglu
        except Exception as exc:  # pragma: no cover - 环境相关
            logger.warning("sdaa.swiglu unavailable, SwiGLU overlay disabled: %s", exc)
            _SWIGLU = None
    return _SWIGLU


def fused_swiglu(gu, intermediate):
    """参考实现：`gu (..., 2I) -> (..., I)`，`silu(前半) * 后半`。

    用于正确性测试与通用调用；含 reshape/视图，模型热路径使用无分支闭包。
    """
    op = _sdaa_swiglu()
    if op is None:
        raise RuntimeError("sdaa.swiglu not available")
    if not gu.is_contiguous():
        gu = gu.contiguous()
    tokens = gu.numel() // gu.shape[-1]
    out = op(gu.reshape(1, tokens, 1, 2 * intermediate), tokens, 1, intermediate)
    return out.reshape(*gu.shape[:-1], intermediate)


def fused_mlp(x, w_gu, down_weight, intermediate):
    """参考实现：合并权重 GEMM + 融合 SwiGLU + down_proj（无 bias，Hy-MT2 语义）。"""
    gu = F.linear(x.reshape(-1, x.shape[-1]), w_gu)
    act = fused_swiglu(gu, intermediate)
    return F.linear(act.reshape(*x.shape[:-1], intermediate), down_weight)


def _make_fused_forward(w_gu, down_proj, hidden, intermediate, op):
    """生成无分支热路径闭包：合并 GEMM -> 融合 swiglu -> down_proj。

    注意：实例属性 `forward` 不经方法绑定，签名只接收 `x`（与 RMSNorm overlay 一致），
    故 `down_proj` 在绑定期闭包捕获。
    """

    def forward(x):
        gu = F.linear(x.reshape(-1, hidden), w_gu)
        tokens = gu.shape[0]
        act = op(gu.view(1, tokens, 1, 2 * intermediate), tokens, 1, intermediate)
        return down_proj(act.view(*x.shape[:-1], intermediate))

    return forward


def apply_swiglu_overlay(model, verbose=True):
    """把 `HunYuanDenseV1MLP.forward` 替换为融合实现，返回绑定模块数。

    只在权重为 bf16 且位于 sdaa 设备上时绑定；其它情况保持原实现。
    幂等：已绑定的模块跳过，不覆盖已保存的原始 forward。
    """
    op = _sdaa_swiglu()
    if op is None:
        if verbose:
            print("[overlay] sdaa.swiglu not available; SwiGLU overlay skipped")
        return 0

    from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
        HunYuanDenseV1MLP,
    )

    bound = 0
    skipped = 0
    for module in model.modules():
        if not isinstance(module, HunYuanDenseV1MLP):
            continue
        if "_hy_fused_swiglu" in module.__dict__:
            bound += 1
            continue
        weight = module.gate_proj.weight
        if weight.dtype != torch.bfloat16 or weight.device.type != "sdaa":
            skipped += 1
            continue
        # 绑定期一次性构建合并权重 [W_gate; W_up] -> (2I, H)，避免 cat 进热路径
        w_gu = torch.cat(
            [module.gate_proj.weight.detach(), module.up_proj.weight.detach()], dim=0
        ).contiguous()
        module._hy_orig_forward = module.forward
        module._hy_fused_forward = _make_fused_forward(
            w_gu, module.down_proj, module.hidden_size, module.intermediate_size, op
        )
        module._hy_fused_swiglu = True
        module.forward = module._hy_fused_forward
        bound += 1

    if verbose:
        print(
            f"[overlay] fused SwiGLU bound on {bound} MLP modules "
            f"(skipped {skipped} non-bf16/sdaa)"
        )
    return bound


def remove_swiglu_overlay(model):
    """移除实例级 forward 覆盖，恢复原实现（用于对照测试）。"""
    from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
        HunYuanDenseV1MLP,
    )

    removed = 0
    for module in model.modules():
        if isinstance(module, HunYuanDenseV1MLP) and "_hy_fused_swiglu" in module.__dict__:
            module.forward = module._hy_orig_forward
            del module.__dict__["_hy_fused_swiglu"]
            module.__dict__.pop("_hy_fused_forward", None)
            module.__dict__.pop("_hy_orig_forward", None)
            removed += 1
    return removed
