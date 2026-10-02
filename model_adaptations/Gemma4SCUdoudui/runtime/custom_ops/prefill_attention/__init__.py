"""Custom SDAA prefill attention (dense KV + torch SDPA)."""
from custom_ops.prefill_attention.op import sdpa_prefill_attention

__all__ = ["sdpa_prefill_attention"]
