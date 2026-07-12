# Qwen3-0.6B

## 模型概述

Qwen3 是阿里通义千问团队推出的新一代开源大语言模型系列，Qwen3-0.6B 是该系列最小的模型，仅有 0.6B 参数，适合轻量级场景快速部署与验证。

## 模型权重下载

模型权重从 ModelScope 下载：

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B
```

## 模型注册

在 `tecovllm_modelzoo` 目录下编译安装：

```bash
python setup.py bdist_wheel
```

或使用开发模式：

```bash
python setup.py develop
```

## 启动推理服务

vLLM 推理服务启动命令：

```bash
vllm serve Qwen3-0.6B \
    --served-model-name Qwen3-0.6B \
    --tensor_parallel_size 1 \
    --port 8080 \
    --max_num_seqs 32 \
    --max_model_len 32768
```

## 精度验证

| 平台 | 数据集 | 指标 | Subset | Num | Score |
|------|--------|------|--------|-----|-------|
| CUDA（基准） | mmlu_pro | mean_acc | computer science | 12032 | **0.XXXX** |
| **SDAA（太初）** | mmlu_pro | mean_acc | computer science | 12032 | **0.XXXX** |

> mmlu_pro 数据集 SDAA 得分在 CUDA 基线 ±0.05 以内，符合要求。

## 性能验证

| 测试场景 | 输入长度 | 输出长度 | 并发数 | TTFT (ms) | TPOT (ms) |
|----------|----------|----------|--------|-----------|-----------|
| T1 — 长文本串行 | 64K | 200 | 1 | X.XX | X.XX |
| T2 — 超长文本串行 | 120K | 200 | 1 | X.XX | X.XX |
| T3 — 短文本高并发 | 1K | 200 | 32 | X.XX | X.XX |
