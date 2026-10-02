# Gemma 4 12B-it (SCU 都队)

This adaptation runs the text generation path of `google/gemma-4-12B-it` on
TecoVLLM/SDAA without modifying vendor site-packages or copying model weights.
The checkpoint remains referenced through `MODEL_ROOT` (default:
`/gpfs/model`).

## Launch

```bash
bash model_adaptations/Gemma4SCUdoudui/run.sh
```

The launch script installs the process-local overlay before vLLM imports its
model and attention modules. It also disables chunked prefill, because the
right-aligned causal mask for that shape is not supported by the SDAA fused
path.

## What is optimized

`runtime/custom_ops/block_attention` binds the attention implementation once
when each `BlockAttentionImpl` is constructed:

- Gemma sliding-window layers (head dimension 256) gather only the trailing
  1024 KV positions and use fused non-causal SDPA for decode.
- Gemma global layers (head dimension 512) use a single-KV-head math path that
  avoids materializing the 16-way GQA repeat.
- KV cache writes still use the vendor `reshape_and_cache` ABI.

The window and head-dimension choices are made during construction; the
forward hot path does not read environment variables or select a backend.
The implementation was first validated on the team `model/gemma-4-12b-it`
branch: the fixed 32-token generation matched the baseline token IDs, D256
and D512 operator comparisons passed, and the CP1 median improved from
14.7759 s (2.098 tok/s) to 13.4920 s (2.298 tok/s).

## Compatibility shims

`overlay/gemma4_shim.py` registers the checkpoint's `gemma4_unified` config,
normalizes the 5.x tokenizer `extra_special_tokens` list for the installed
Transformers version, and skips the checkpoint's vision/audio-only tensors
that are not consumed by the text model.

## Runtime contract

Set `MODEL_ROOT` and, when needed, `GEMMA_MAX_MODEL_LEN` before launch. The
default served name is `Gemma-4-12B-it`; `GEMMA_PORT`, `GEMMA_HOST`,
`GEMMA_UTIL`, and `GEMMA_MAX_BATCHED_TOKENS` override the corresponding
server values. A working `teco-ops` wheel providing `reshape_and_cache` and
the SDAA runtime is required.
