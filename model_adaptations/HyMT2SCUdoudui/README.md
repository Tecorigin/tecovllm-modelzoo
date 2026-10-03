# Hy-MT2-1.8B (SCU 都队)

This adaptation keeps the checkpoint under the read-only `MODEL_ROOT` tree and
installs a process-local overlay before TecoVLLM builds the model.  The overlay
targets the classes used by `vllm.model_executor.models.hunyuan_v1`; it does not
modify Transformers, vLLM, or the installed vendor packages.

## Launch

```bash
MODEL_ROOT=/gpfs/model bash model_adaptations/HyMT2SCUdoudui/run.sh
```

The launch contract uses the required served name, an explicit port,
`--trust-remote-code`, and `--no-enable-prefix-caching`.  `HY_MT2_MODEL`,
`HY_MT2_PORT`, `HY_MT2_TP_SIZE`, and `HY_MT2_MAX_MODEL_LEN` override the local
defaults.

## Operator overlay

`runtime/overlay/sitecustomize.py` binds two model-side optimizations during
process initialization:

- `vllm`'s `RMSNorm` is routed to the vendor `tecoops.rms_norm` ABI.  The
  runtime path is FP16, matching the measured vendor contract.
- `HunYuanMLP` keeps the existing merged gate/up projection and replaces the
  activation with `torch.ops.sdaa.swiglu`.  This avoids the extra `cat` and
  preserves the model's projection shapes.

The branch-level evidence is recorded in the team repository under
`experiments/hy-mt2-1.8b.md`: RMSNorm and SwiGLU each passed the real-shape
correctness tests, fixed 32-token output matched the baseline, and the
SwiGLU device-time A/B was 3.16% lower.  The original Transformers overlays
remain available in the model branch for their independent tests; this bundle
uses the vLLM classes so the integration point is explicit.

The shared Teco-Ops `flash_attn_varlen_func`, `reshape_and_cache`, and
`rms_norm` PRs remain separate dependencies of the SDAA runtime.  This PR only
contains the Hy-MT2 model-side bindings and does not copy vendor kernels.
