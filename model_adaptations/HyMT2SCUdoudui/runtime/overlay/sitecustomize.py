"""Process-init overlays for the Hy-MT2 vLLM implementation.

The model branch was validated first through Transformers.  TecoVLLM uses its
own ``hunyuan_v1`` implementation, so the process-local overlay binds the
same two measured optimizations to the vLLM classes instead of importing the
Transformers modules.  The vendor installation remains untouched.
"""

import sys


def install() -> None:
    try:
        import torch
        import torch.nn as nn
        import torch_sdaa  # noqa: F401
        import torch_sdaa.nn  # noqa: F401  registers torch.ops.sdaa.swiglu
        import tecoops
        import vllm.model_executor.models.hunyuan_v1 as hunyuan
    except Exception as exc:  # pragma: no cover - depends on the SDAA image
        raise RuntimeError(f"Hy-MT2 overlay dependencies unavailable: {exc}") from exc

    if not hasattr(tecoops, "rms_norm"):
        raise RuntimeError("Hy-MT2 overlay requires tecoops.rms_norm")
    swiglu = getattr(torch.ops.sdaa, "swiglu", None)
    if swiglu is None:
        raise RuntimeError("Hy-MT2 overlay requires torch.ops.sdaa.swiglu")

    class TecoopsRMSNorm(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
            var_hidden_size: int | None = None,
            has_weight: bool = True,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            if var_hidden_size not in (None, hidden_size):
                raise ValueError("Hy-MT2 RMSNorm overlay does not support var_hidden_size")
            self.hidden_size = hidden_size
            self.variance_epsilon = eps
            self.has_weight = has_weight
            if has_weight:
                self.weight = nn.Parameter(
                    torch.ones(hidden_size, dtype=dtype or torch.get_default_dtype())
                )

        def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
            x2 = x.reshape(-1, x.shape[-1])
            residual2 = residual.reshape_as(x2) if residual is not None else None
            out2 = torch.empty_like(x2)
            residual_out2 = torch.empty_like(x2) if residual2 is not None else None
            tecoops.rms_norm(
                x2,
                self.weight.data if self.has_weight else None,
                residual2,
                out2,
                residual_out2,
                float(self.variance_epsilon),
            )
            out = out2.reshape_as(x)
            if residual_out2 is not None:
                return out, residual_out2.reshape_as(x)
            return out

    def fused_mlp_forward(self, x: torch.Tensor):
        gate_up, _ = self.gate_up_proj(x)
        tokens = gate_up.numel() // gate_up.shape[-1]
        intermediate = gate_up.shape[-1] // 2
        activated = swiglu(
            gate_up.reshape(1, tokens, 1, gate_up.shape[-1]),
            tokens,
            1,
            intermediate,
        ).reshape(*gate_up.shape[:-1], intermediate)
        output, _ = self.down_proj(activated)
        return output

    hunyuan.RMSNorm = TecoopsRMSNorm
    hunyuan.HunYuanMLP.forward = fused_mlp_forward
    if hunyuan.RMSNorm is not TecoopsRMSNorm:
        raise RuntimeError("Hy-MT2 RMSNorm binding failed")
    if hunyuan.HunYuanMLP.forward is not fused_mlp_forward:
        raise RuntimeError("Hy-MT2 SwiGLU binding failed")
    print("[hy-mt2-overlay] RMSNorm and SwiGLU bindings installed", file=sys.stderr)


install()
