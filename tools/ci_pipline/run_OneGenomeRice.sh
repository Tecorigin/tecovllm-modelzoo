#!/bin/bash
# vllm serve /tecogpfs/models/OpenGVLab/InternVL3_5-8B/ \
#     --trust_remote_code \
#     --tensor-parallel-size 4 \
#     --port 8222 \
#     --max-num-seqs 32 \
#     --served-model-name InternVL3_5-8B

vllm serve /tecogpfs/models/zhejianglab/OneGenomeRice \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --port 8222 \
    --max-num-seqs 32 \
    --served-model-name OneGenomeRice \
    --no-enable-prefix-caching
