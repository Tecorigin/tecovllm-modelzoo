"""Qwen3 语言塔 RMSNorm 的**官方 tecoops 绑定**（InternVL3_5-8B）。

背景与目标
----------
审计实测：InternVL3_5-8B 推理**从未调用**官方 ``tecoops.rms_norm``
（全量 kernel 表 0 命中，见 ``experiments/internvl3_5-8b.md`` §12.2）。
原因是 vLLM 的 Qwen3 实现自带 ``vllm.model_executor.layers.layernorm.RMSNorm``，
而本分支此前只移植了 BlockAttention 的 overlay，没有移植 MiniCPM 的
Llama RMSNorm 绑定。本文件把 **Qwen3 语言塔** 的 RMSNorm 绑到官方算子上。

绑定点（实测自 vLLM 0.20.2.dev0 源码，不是猜测）
------------------------------------------------
* ``vllm/model_executor/models/qwen3.py``
  - ``Qwen3Attention``：``self.q_norm`` / ``self.k_norm``（``head_dim``，第 142/143 行，
    forward 第 152-158 行**无条件**使用 —— Qwen3 有 qk-norm）；
  - ``Qwen3DecoderLayer``：``input_layernorm`` / ``post_attention_layernorm``（第 211/212 行）。
* ``vllm/model_executor/models/qwen2.py``
  - ``Qwen2Model``：最终 ``self.norm``（第 415 行）—— ``Qwen3Model(Qwen2Model)``
    **继承**该类，所以最终 norm 用的是 ``qwen2`` 模块的全局名字，必须一并替换，
    否则会漏掉每步 1 次调用（这一步以前很容易被忽略）。

因此每层 4 次 + 每步 1 次最终 norm = **每 forward 145 次**（36 层 ⇒ 36 × 4 + 1）。

热路径契约（AGENTS.md 第 6 条）
------------------------------
* 后端选择只在 ``install_qwen3_rms_norm_overlay()`` 里做一次；装不上就 ``sys.exit(1)``
  （fail-closed，**没有 CPU 回退、没有 fallback 分支、没有 getenv**）。
* ``TecoopsRMSNorm.forward`` 里没有任何条件分支：两个不透明算子
  （无残差 / 带残差）在**类工厂**里就已经绑定好，forward 只按调用形态分派，
  这是数据形态而非后端选择，且与 vLLM 原实现一一对应。
* 官方 pybind 函数用 ``torch.library.custom_op`` 包成**不透明算子**：
  ``torch.compiler.allow_in_graph`` 挡不住 Dynamo 内联，一旦内联就会在 FakeTensor 上
  调 ``data_ptr()`` 炸掉（MiniCPM 分支实测踩过）。这里显式 ``mutates_args`` + fake impl。

参考实现（``reference_rms_norm``）**只给离线单测用，不是运行时回退**。
"""

import os
import sys

import torch
from torch import nn

from custom_ops.receipt import bind_official_api

_NAMESPACE = "internvl_overlay"
_OPS_CACHE: "tuple | None" = None

#: 本模型实测结构：36 层 × 4 个 RMSNorm + 1 个最终 norm
RMS_NORMS_PER_LAYER = 4
FINAL_NORMS_PER_FORWARD = 1


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: "torch.Tensor | None" = None,
):
    """fp32 参考实现（**离线单测专用**，绝不是运行时回退）。

    语义与 vLLM 的 ``RMSNorm.forward_static`` 一致：
    ``residual_out = x + residual``（精确），``out = rms(residual_out) * weight``。
    """
    if residual is not None:
        x = (x.float() + residual.float()).to(x.dtype)
        residual_out = x
    else:
        residual_out = None

    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    out = (x_fp32 * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)
    if residual is None:
        return out
    return out, residual_out


def register_opaque_rms_norm(kernel) -> tuple:
    """把官方 pybind ``kernel`` 注册成两个 compile-safe 的不透明算子。

    返回 ``(rms_norm, rms_norm_add)``，两者都**只写输出、返回 None**。
    进程内只注册一次（``torch.library`` 重复注册会报错）。
    """
    global _OPS_CACHE
    if _OPS_CACHE is not None:
        return _OPS_CACHE

    @torch.library.custom_op(f"{_NAMESPACE}::rms_norm", mutates_args=("out",))
    def rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        eps: float,
    ) -> None:
        kernel(
            x.reshape(-1, x.shape[-1]),
            weight,
            None,
            out.reshape(-1, out.shape[-1]),
            None,
            float(eps),
        )

    @rms_norm.register_fake
    def _rms_norm_fake(x, weight, out, eps):
        return None

    @torch.library.custom_op(
        f"{_NAMESPACE}::rms_norm_add", mutates_args=("out", "residual_out")
    )
    def rms_norm_add(
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        out: torch.Tensor,
        residual_out: torch.Tensor,
        eps: float,
    ) -> None:
        x2 = x.reshape(-1, x.shape[-1])
        kernel(
            x2,
            weight,
            residual.reshape_as(x2),
            out.reshape(-1, out.shape[-1]),
            residual_out.reshape(-1, residual_out.shape[-1]),
            float(eps),
        )

    @rms_norm_add.register_fake
    def _rms_norm_add_fake(x, weight, residual, out, residual_out, eps):
        return None

    _OPS_CACHE = (rms_norm, rms_norm_add)
    return _OPS_CACHE


def make_rms_norm_class(rms_norm_op, rms_norm_add_op, vendor_cls=None, min_hidden=None):
    """生成 `TecoopsRMSNorm` 类，签名与返回契约对齐 vLLM 的 `RMSNorm`。

    ``min_hidden`` 给定即进入**收窄模式**：``hidden_size < min_hidden`` 的模块
    （本模型里是 q/k norm，``head_dim=128``）**不**走官方算子，而是把厂商原生实现
    的方法直接绑到**实例**上。决定发生在**模块构造期**（进程初始化的一部分），
    ``forward`` 里没有任何 ``if``：收窄实例靠实例属性拿到厂商的 ``forward``。

    返回契约（必须与 vLLM 一致，否则 Qwen3 的 ``hidden_states, residual = norm(...)``
    解包会错）：
    * ``residual is None`` → 返回 ``out``
    * 否则 → 返回 ``(out, residual_out)``，其中 ``residual_out == x + residual``

    ``has_weight`` 一类额外参数被忽略：vLLM 的 Qwen3/Qwen2 从不传 ``has_weight=False``，
    而「不缩放」在数学上等价于 ``weight = 1``，与本类默认值一致。
    """

    class TecoopsRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs: object) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.variance_epsilon = eps
            if min_hidden is not None and hidden_size < min_hidden:
                # 收窄：委托给厂商原生实现。**只共享同一个 weight Parameter**
                # （用 object.__setattr__ 把厂商模块挂在模块表之外，避免 state_dict
                # 出现重复 key），并把 forward 直接绑成厂商实例的方法 ⇒ 热路径零分支。
                vendor = vendor_cls(hidden_size, eps=eps, **kwargs)
                object.__setattr__(self, "_vendor_impl", vendor)
                self.weight = vendor.weight
                self.forward = vendor.forward
                return
            # vLLM 在模型构造期已把默认 dtype 设为模型 dtype（fp16），
            # 这里不写死 dtype，免得与 q/k norm 的 head_dim 场景不一致。
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x: torch.Tensor, residual: "torch.Tensor | None" = None):
            x2 = x.reshape(-1, x.shape[-1])
            out2 = torch.empty_like(x2)
            eps = float(self.variance_epsilon)
            if residual is None:
                rms_norm_op(x2, self.weight.data, out2, eps)
                return out2.reshape_as(x)
            res2 = residual.reshape_as(x2)
            residual_out2 = torch.empty_like(x2)
            rms_norm_add_op(x2, self.weight.data, res2, out2, residual_out2, eps)
            return out2.reshape_as(x), residual_out2.reshape_as(x)

    TecoopsRMSNorm.__name__ = "TecoopsRMSNorm"
    TecoopsRMSNorm.__qualname__ = "TecoopsRMSNorm"
    TecoopsRMSNorm.__doc__ = (
        "Qwen3 RMSNorm bound to the official tecoops.rms_norm ABI at process init."
    )
    return TecoopsRMSNorm


def _fatal(message: str) -> None:
    sys.stderr.write(f"FATAL: [custom_ops] {message}\n")
    sys.exit(1)


def install_qwen3_rms_norm_overlay():
    """把 Qwen3 语言塔的 ``RMSNorm`` 名字绑到官方 ``tecoops.rms_norm``。

    fail-closed：任何一步不满足预期都 ``sys.exit(1)``，绝不静默降级。
    返回绑定后的类（便于测试与诊断）。
    """
    try:
        import tecoops
    except Exception as exc:
        _fatal(f"Failed to import 'tecoops' for the RMSNorm overlay: {exc}")

    kernel = getattr(tecoops, "rms_norm", None)
    if kernel is None:
        _fatal("tecoops.rms_norm is unavailable; refusing to install a fallback")

    try:
        from vllm.model_executor.layers.layernorm import RMSNorm as vendor_rms_norm
        import vllm.model_executor.models.qwen2 as qwen2_module
        import vllm.model_executor.models.qwen3 as qwen3_module
    except Exception as exc:
        _fatal(f"Failed to import the Qwen2/Qwen3 RMSNorm binding targets: {exc}")

    # 前置身份校验：两个模块必须都还从 layernorm 模块导入同一个类。
    for module in (qwen3_module, qwen2_module):
        current = getattr(module, "RMSNorm", None)
        if current is not vendor_rms_norm:
            _fatal(
                "Pre-install identity check failed: "
                f"{module.__name__}.RMSNorm is {current!r}, "
                f"expected vllm.model_executor.layers.layernorm.RMSNorm"
            )

    # 最终 norm 在 Qwen2Model 里构造；Qwen3Model 必须仍然是它的子类。
    if not issubclass(qwen3_module.Qwen3Model, qwen2_module.Qwen2Model):
        _fatal(
            "Qwen3Model no longer subclasses Qwen2Model; the final RMSNorm binding "
            "target has moved and this overlay would silently miss it"
        )

    min_hidden_raw = os.environ.get("INTERNVL_RMS_NORM_MIN_HIDDEN")
    min_hidden = int(min_hidden_raw) if min_hidden_raw else None

    ops = register_opaque_rms_norm(kernel)
    # 收据名分开记：无残差 / 带残差两类调用的结构式不同，合起来就丢掉了对账能力。
    bound_norm = bind_official_api(ops[0], "tecoops.rms_norm")
    bound_norm_add = bind_official_api(ops[1], "tecoops.rms_norm_add")
    tecoops_rms_norm = make_rms_norm_class(
        bound_norm, bound_norm_add, vendor_cls=vendor_rms_norm, min_hidden=min_hidden
    )

    try:
        qwen3_module.RMSNorm = tecoops_rms_norm
        qwen2_module.RMSNorm = tecoops_rms_norm
    except Exception as exc:
        _fatal(f"Failed to bind TecoopsRMSNorm onto the Qwen3/Qwen2 modules: {exc}")

    # 安装后身份校验
    for module in (qwen3_module, qwen2_module):
        if getattr(module, "RMSNorm", None) is not tecoops_rms_norm:
            _fatal(
                "Post-install identity check failed: "
                f"{module.__name__}.RMSNorm is not TecoopsRMSNorm"
            )

    sys.stderr.write(
        "[custom_ops] Qwen3/Qwen2 RMSNorm overlay bound to tecoops.rms_norm "
        f"(device kernels: {getattr(kernel, '__module__', 'tecoops')}, "
        f"{RMS_NORMS_PER_LAYER} per layer + {FINAL_NORMS_PER_FORWARD} final per forward, "
        f"narrow_min_hidden={min_hidden}).\n"
    )
    return tecoops_rms_norm
