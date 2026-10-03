"""Process initialization overlay for TecoVLLM on SDAA accelerator.

Binds custom_ops adaptations before engine initialization to avoid
hot-path branching and keep vendor site-packages strictly read-only.
"""

import os
import sys

_RECEIPT_PATH = os.environ.get("TECOOPS_CALL_RECEIPT")
_RMS_CALL_RECORDED = False


def install_llama_rms_norm_overlay() -> None:
    """Bind Llama's RMSNorm construction to the official tecoops API."""
    try:
        import torch
        import torch.nn as nn
        import vllm.model_executor.models.llama as llama_model
        import tecoops
    except Exception as exc:
        sys.stderr.write(f"FATAL: [custom_ops] Failed to import RMSNorm overlay dependencies: {exc}\n")
        sys.exit(1)

    if not hasattr(tecoops, "rms_norm"):
        sys.stderr.write("FATAL: [custom_ops] tecoops.rms_norm is unavailable\n")
        sys.exit(1)

    # The official wheel exposes pybind functions. Register them once during
    # process initialization so vLLM's normal compilation mode treats the
    # calls as opaque graph-safe operators instead of graph-breaking on a
    # third-party extension.
    for name in ("rms_norm", "reshape_and_cache", "flash_attn_varlen_func"):
        fn = getattr(tecoops, name, None)
        if fn is None:
            sys.stderr.write(f"FATAL: [custom_ops] tecoops.{name} is unavailable\n")
            sys.exit(1)
        try:
            torch.compiler.allow_in_graph(fn)
        except AttributeError:
            torch._dynamo.allow_in_graph(fn)

    class TecoopsRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6, **_: object) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.variance_epsilon = eps
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.get_default_dtype()))

        def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
            global _RMS_CALL_RECORDED
            if _RECEIPT_PATH and not _RMS_CALL_RECORDED:
                _RMS_CALL_RECORDED = True
                with open(_RECEIPT_PATH, "a", encoding="utf-8") as receipt:
                    receipt.write("CALL tecoops.rms_norm\n")
            x2 = x.reshape(-1, x.shape[-1])
            residual2 = residual.reshape_as(x2) if residual is not None else None
            out2 = torch.empty_like(x2)
            residual_out2 = torch.empty_like(x2) if residual2 is not None else None
            tecoops.rms_norm(
                x2,
                self.weight.data,
                residual2,
                out2,
                residual_out2,
                float(self.variance_epsilon),
            )
            out = out2.reshape_as(x)
            if residual_out2 is not None:
                return out, residual_out2.reshape_as(x)
            return out

    llama_model.RMSNorm = TecoopsRMSNorm
    sys.stderr.write("[custom_ops] Llama RMSNorm overlay bound to tecoops.rms_norm.\n")


def install_block_attention_overlay() -> None:
    """Installs the SDAA BlockAttention forward overlay.

    Fails closed: any import error, missing target attribute, assignment
    failure, or identity check failure immediately terminates the process
    with a non-zero exit code.
    """
    try:
        import vllm_sdaa.attention.block_attn as block_attn_module
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to import target module 'vllm_sdaa.attention.block_attn': {exc}\n"
        )
        sys.exit(1)

    if not hasattr(block_attn_module, "BlockAttentionImpl"):
        sys.stderr.write(
            "FATAL: [custom_ops] Target class 'BlockAttentionImpl' not found in 'vllm_sdaa.attention.block_attn'\n"
        )
        sys.exit(1)

    impl_cls = block_attn_module.BlockAttentionImpl
    if not hasattr(impl_cls, "forward"):
        sys.stderr.write(
            "FATAL: [custom_ops] Target attribute 'forward' not found in 'BlockAttentionImpl'\n"
        )
        sys.exit(1)

    try:
        from custom_ops.block_attention.op import sdaa_block_attention_forward
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to import 'sdaa_block_attention_forward': {exc}\n"
        )
        sys.exit(1)

    try:
        impl_cls.forward = sdaa_block_attention_forward
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to bind forward method onto BlockAttentionImpl: {exc}\n"
        )
        sys.exit(1)

    if getattr(impl_cls, "forward", None) is not sdaa_block_attention_forward:
        sys.stderr.write(
            "FATAL: [custom_ops] Post-install identity check failed: BlockAttentionImpl.forward is not sdaa_block_attention_forward\n"
        )
        sys.exit(1)

    sys.stderr.write("[custom_ops] BlockAttention overlay bound successfully.\n")


install_llama_rms_norm_overlay()
install_block_attention_overlay()
