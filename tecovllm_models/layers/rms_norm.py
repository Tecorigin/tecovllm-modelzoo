import torch
import torch.nn as nn

def rms_norm(
    x: torch.Tensor,
    variance_epsilon: float,
    orig_dtype: torch.dtype,
    weight: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual
        residual = x.to(orig_dtype)

    x_var = x
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)

    x = x * torch.rsqrt(variance + variance_epsilon)
    x = x.to(orig_dtype)
    if weight is not None:
        x = x * weight
    if residual is None:
        return x
    else:
        return x, residual

class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        weight_dtype = dtype or torch.get_default_dtype()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return rms_norm(
            x,
            self.variance_epsilon,
            x.dtype,
            self.weight.data,
            residual,
        )
