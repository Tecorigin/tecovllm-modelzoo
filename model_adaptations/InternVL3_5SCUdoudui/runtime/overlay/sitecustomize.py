"""Process initialization overlay for TecoVLLM on SDAA accelerator.

Binds the in-repo BlockAttention compatibility layer before engine
initialization so that ``vllm_sdaa``'s unimplemented attention stub is
replaced without ever writing to vendor site-packages and without leaving a
per-forward branch in the hot path.

It also binds the Qwen3 language tower's ``RMSNorm`` to the official
``tecoops.rms_norm`` operator.  Audit fact that motivated it: before this
overlay the official operator had **0 calls** in a real forward (see
``experiments/internvl3_5-8b.md`` §12.2).  The binding replaces the module-level
``RMSNorm`` name in ``vllm.model_executor.models.qwen3`` **and**
``...models.qwen2`` (the final norm is constructed by ``Qwen2Model``, which
``Qwen3Model`` inherits).  It is installed unconditionally and fails closed: if
the vendor layout moved, the process exits non-zero instead of silently
running without the official operator.

Scope note (InternVL3_5-8B): the MiniCPM5-1B branch's *Llama-specific* binding
is not ported verbatim; this one targets Qwen3 and is validated by an
init-time identity check plus a call-count receipt.

Loaded automatically by CPython via ``sitecustomize`` when this directory is
placed on ``PYTHONPATH`` (see scripts/serve_internvl.sh).
"""

import sys


def install_rms_norm_overlay() -> None:
    """Bind the Qwen3 language tower RMSNorm to official tecoops.rms_norm."""
    try:
        from custom_ops.rms_norm.op import install_qwen3_rms_norm_overlay
    except Exception as exc:
        sys.stderr.write(
            f"FATAL: [custom_ops] Failed to import the RMSNorm overlay installer: {exc}\n"
        )
        sys.exit(1)

    # The installer itself is fail-closed (sys.exit on every unmet precondition).
    install_qwen3_rms_norm_overlay()


def install_block_attention_overlay() -> None:
    """Install the SDAA BlockAttention forward overlay.

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


install_rms_norm_overlay()
install_block_attention_overlay()
