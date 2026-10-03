# InternVL3_5-8B (SCU 都队)

This adaptation serves `OpenGVLab/InternVL3_5-8B` on SDAA through TecoVLLM.
Weights remain read-only under `MODEL_ROOT`; no vendor site-package or model
copy is required.

## Launch

```bash
MODEL_ROOT=/gpfs/model bash model_adaptations/InternVL3_5SCUdoudui/run.sh
```

The default contract uses FP16, tensor parallel size 2, an explicit port,
`--trust-remote-code`, `--no-enable-prefix-caching`, and
`--no-enable-chunked-prefill`.  Override the local path and service settings
with `INTERNVL_MODEL`, `INTERNVL_PORT`, `INTERNVL_TP_SIZE`, and
`INTERNVL_MAX_MODEL_LEN`.

## Operator path

The process-init overlay binds the Qwen3 language tower's RMSNorm to the
official `tecoops.rms_norm` ABI and replaces the unimplemented SDAA
`BlockAttentionImpl.forward`.  The attention adapter writes paged KV cache via
`reshape_and_cache`, gathers complete per-request K/V blocks, and dispatches
prefill through the fused SDAA SDPA path.  The official flash ABI wrapper is
kept in the bundle for the model's operator contract and independent checks.

The source branch `model/internvl3_5-8b` contains real-shape hardware evidence
for the RMSNorm, reshape/cache, and prefill paths, plus fixed 32-token text and
natural-image smoke logs.  The prefill adapter explicitly disables chunked
prefill because SDAA's non-square causal mask path is not semantically safe.
Official benchmark and accuracy jobs remain the repository's external
evaluation step.
