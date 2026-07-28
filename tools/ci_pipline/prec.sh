#!/bin/bash

# ============================================================
# prec.sh — vLLM 单模型精度测试脚本
# 用法: ./prec.sh <model_name> [ip] [port]
#
#   model_name: Hy-MT2-1.8B | MiniCPM5-1B | InternVL3_5-8B | gemma-4-12B-it
#   ip:         vLLM 服务 IP（默认 0.0.0.0）
#   port:       vLLM 服务端口（默认 8000）
#
# 前提: vLLM 服务已启动，evalscope 已安装
# ============================================================

export HF_ENDPOINT=https://hf-mirror.com
MODEL_NAME="${1:?Usage: $0 <model_name> [ip] [port]}"
IP="${2:-0.0.0.0}"
PORT="${3:-8000}"

LOG_DIR="./prec_logs/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# ---- 检查 evalscope 是否可用 ----
if ! command -v evalscope &>/dev/null; then
    echo "错误: evalscope 未找到，请确认已安装并激活正确的 conda 环境" >&2
    exit 1
fi

API_URL="http://${IP}:${PORT}/v1/chat/completions"

# ---- 辅助函数 ----
run_eval() {
    local dataset="$1"
    local ds_args="$2"
    local gen_cfg="$3"
    local log="$LOG_DIR/${dataset}.log"

    echo ">>> [${MODEL_NAME}] 开始评测 (${dataset}) ..."
    echo "    API: ${API_URL}"

    evalscope eval \
        --model "$MODEL_NAME" \
        --api-url "$API_URL" \
        --api-key EMPTY_TOKEN \
        --datasets "$dataset" \
        --dataset-args "$ds_args" \
        --eval-batch-size 8 \
        --limit 200 \
        --generation-config "$gen_cfg" \
        --ignore-errors \
        2>&1 | tee "$log"

    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "!!! [${MODEL_NAME}] ${dataset} 评测失败 (exit=${rc})" >&2
        return "$rc"
    fi
    echo ">>> [${MODEL_NAME}] ${dataset} 评测完成"
}

# ============================================================
# 主流程
# ============================================================
echo "===== 精度测试: ${MODEL_NAME} ====="
echo "  服务地址: ${API_URL}"
echo "  日志目录: ${LOG_DIR}"
echo "===================================="

rc=0

case "$MODEL_NAME" in
    Hy-MT2-1.8B)
        run_eval "wmt24pp" \
            '{"wmt24pp": {"subset_list": ["en-zh_cn"], "metric_list": [{"comet": {"model_id_or_path": "evalscope/wmt22-comet-da"}}]}}' \
            'do_sample=true,temperature=0.7,top_p=0.6,top_k=20,repetition_penalty=1.05,max_tokens=1024,timeout=3600' \
            || rc=1
        ;;
    MiniCPM5-1B)
        run_eval "mmlu_pro" \
            '{"mmlu_pro": {"subset_list": ["computer science"]}}' \
            '{"do_sample":true,"temperature":0.9,"top_p":0.95,"max_tokens":1024,"timeout":7200,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
            || rc=1
        ;;
    InternVL3_5-8B)
        run_eval "mmlu_pro" \
            '{"mmlu_pro": {"subset_list": ["computer science"]}}' \
            'do_sample=true,temperature=0.6,max_tokens=1024,timeout=3600' \
            || rc=1
        ;;
    gemma-4-12B-it)
        run_eval "mmlu_pro" \
            '{"mmlu_pro": {"subset_list": ["computer science"]}}' \
            'do_sample=true,temperature=1.0,top_p=0.95,top_k=64,max_tokens=4096,timeout=3600' \
            || rc=1
        ;;
    *)
        echo "错误: 未知模型 '$MODEL_NAME'" >&2
        echo "支持: Hy-MT2-1.8B, MiniCPM5-1B, InternVL3_5-8B, gemma-4-12B-it" >&2
        exit 1
        ;;
esac

echo "===== 精度测试完成: ${MODEL_NAME}，日志目录: $LOG_DIR ====="
exit "$rc"
