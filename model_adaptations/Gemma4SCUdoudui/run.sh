#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT="${MODEL_ROOT:-/gpfs/model}"
GEMMA_PATH="${GEMMA_PATH:-${MODEL_ROOT}/google/gemma-4-12B-it}"
PORT="${GEMMA_PORT:-8000}"
HOST="${GEMMA_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${GEMMA_MAX_MODEL_LEN:-8192}"
MAX_BATCHED="${GEMMA_MAX_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
UTIL="${GEMMA_UTIL:-0.92}"

# Bind the reviewed implementation before vLLM constructs its attention
# classes. No vendor package is modified and no model weights are copied.
export GEMMA4_OVERLAY=1
export PYTHONPATH="${SCRIPT_DIR}/overlay:${SCRIPT_DIR}/runtime${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec vllm serve "${GEMMA_PATH}" \
    --served-model-name Gemma-4-12B-it \
    --tensor-parallel-size "${GEMMA_TP_SIZE:-2}" \
    --gpu-memory-utilization "${UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_BATCHED}" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --hf-overrides '{"architectures":["Gemma4ForCausalLM"]}' \
    --dtype float16 \
    --host "${HOST}" \
    --port "${PORT}"
