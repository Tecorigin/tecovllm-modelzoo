#!/bin/bash
vllm serve /tecogpfs/models/Qwen/Qwen3-0.6B/ \
    --tensor-parallel-size 4 \
    --port 8222 \
    --max-num-seqs 32 \
    --hf-override '{"architectures": ["Qwen3DenseForCausalLM"]}'
