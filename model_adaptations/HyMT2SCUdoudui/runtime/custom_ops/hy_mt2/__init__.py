"""Hy-MT2-1.8B 重点算子接入（SDAA）。

两个算子均通过启动期 overlay 注入，不修改任何 site-packages / torch / transformers 文件：

- `rmsnorm_fused`：把 transformers 原生 `HunYuanDenseV1RMSNorm.forward` 替换为
  厂商融合算子 `tecoops.rms_norm` 调用。
- `swiglu_fused`：把 `HunYuanDenseV1MLP.forward` 替换为「合并 gate/up 权重单 GEMM
  + 厂商融合算子 `torch.ops.sdaa.swiglu`」实现。
"""

from .rmsnorm_fused import apply_rmsnorm_overlay, fused_rms_norm, remove_rmsnorm_overlay
from .swiglu_fused import (
    apply_swiglu_overlay,
    fused_mlp,
    fused_swiglu,
    remove_swiglu_overlay,
)

__all__ = [
    "apply_rmsnorm_overlay",
    "remove_rmsnorm_overlay",
    "fused_rms_norm",
    "apply_swiglu_overlay",
    "remove_swiglu_overlay",
    "fused_swiglu",
    "fused_mlp",
]
