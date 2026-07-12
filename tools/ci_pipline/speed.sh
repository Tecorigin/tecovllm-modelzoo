#!/bin/bash

# ============================================================
# speed.sh — vLLM 模型性能测试脚本
# 用法: ./speed.sh <model_name> <tokenizer_path> [ip] [port]
#
#   ip:   vLLM 服务 IP（默认 0.0.0.0）
#   port: vLLM 服务端口（默认 8000）
# ============================================================

MODEL_NAME="${1:?Usage: $0 <model_name> <tokenizer_path> [ip] [port]}"
TOKENIZER_PATH="${2:?}"
IP="${3:-0.0.0.0}"
PORT="${4:-8000}"

LOG_DIR="./speed_logs/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

if ! command -v evalscope &>/dev/null; then
    echo "错误: evalscope 未找到" >&2
    exit 1
fi

API_URL="http://${IP}:${PORT}/v1/chat/completions"

# ---- 辅助函数 ----
run_text() {
    local name="$1" prompt_len="$2" parallel="$3" number="$4"
    local log="$LOG_DIR/${name}.log"
    echo ">>> [${name}] bs=${parallel} ${prompt_len}→200 ..."
    set +e
    evalscope perf \
        --model "$MODEL_NAME" \
        --url "$API_URL" \
        --api openai \
        --tokenizer-path "$TOKENIZER_PATH" \
        --dataset random \
        --min-prompt-length "$prompt_len" --max-prompt-length "$prompt_len" \
        --max-tokens 200 --min-tokens 200 \
        --extra-args '{"ignore_eos": true}' \
        --parallel "$parallel" --number "$number" \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "!!! [${name}] 失败 (exit=${rc})" >&2
    else
        echo ">>> [${name}] 完成"
    fi
    return "$rc"
}

run_image() {
    local name="$1" width="$2" height="$3" img_num="$4" parallel="$5" number="$6"
    local log="$LOG_DIR/${name}.log"
    echo ">>> [${name}] bs=${parallel} ${img_num}×${width}²→200 ..."
    set +e
    evalscope perf \
        --model "$MODEL_NAME" \
        --url "$API_URL" \
        --api openai \
        --tokenizer-path "$TOKENIZER_PATH" \
        --dataset random_vl \
        --max-prompt-length 200 --min-prompt-length 200 \
        --image-width "$width" --image-height "$height" \
        --image-num "$img_num" --image-format RGB \
        --max-tokens 200 --min-tokens 200 \
        --extra-args '{"ignore_eos": true}' \
        --parallel "$parallel" --number "$number" \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "!!! [${name}] 失败 (exit=${rc})" >&2
    else
        echo ">>> [${name}] 完成"
    fi
    return "$rc"
}

# ============================================================
echo "===== 性能测试: ${MODEL_NAME} ====="
echo "  服务地址: ${API_URL}"
echo "  日志目录: ${LOG_DIR}"
echo "===================================="

# ---- Warmup ----
echo ">>> Warmup (1K→200, bs=1) ..."
evalscope perf \
    --model "$MODEL_NAME" \
    --url "$API_URL" \
    --api openai \
    --tokenizer-path "$TOKENIZER_PATH" \
    --dataset random \
    --min-prompt-length 1024 --max-prompt-length 1024 \
    --max-tokens 200 --min-tokens 200 \
    --extra-args '{"ignore_eos": true}' \
    --parallel 1 --number 1 \
    2>&1 | tee "$LOG_DIR/warmup.log" || true
echo ">>> Warmup 完成"

rc=0

# ============================================================
# 按模型执行文本测试
# ============================================================
case "$MODEL_NAME" in
    InternVL3_5-8B)
        run_text "T1_16K_b1"   16384 1  10  || rc=1
        run_text "T2_32K_b1"   32768 1  10  || rc=1
        run_text "T3_1K_b32"   1024  32 320 || rc=1
        # 多模态图片测试
        run_image "I1_5img_b1"         1024 1024 10  1  10  || rc=1
        run_image "I2_2img_b1"         2048 2048 5  1  10  || rc=1
        run_image "I3_1img_b32"        1024 1024 1  32 320 || rc=1
        ;;
    gemma-4-12B-it)
        run_text "T1_64K_b1"   65536  1  10  || rc=1
        run_text "T2_120K_b1" 122880  1  10  || rc=1
        run_text "T3_1K_b32"   1024  32 320  || rc=1
        run_image "I1_multi_img_b1"    1024 1024 10 1  10  || rc=1
        run_image "I2_large_img_b1"    2048 2048 5  1  10  || rc=1
        run_image "I3_single_img_b32"  1024 1024 1  32 320 || rc=1
        ;;
    *)
        # 纯文本模型
        run_text "T1_64K_b1"   65536  1  10  || rc=1
        run_text "T2_120K_b1" 122880  1  10  || rc=1
        run_text "T3_1K_b32"   1024  32 320  || rc=1
        ;;
esac

echo "===== 性能测试完成: ${MODEL_NAME}，日志目录: $LOG_DIR ====="
exit "$rc"
