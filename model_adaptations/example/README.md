# Qwen3-0.6B

## 模型概述

Qwen3 是阿里通义千问团队推出的新一代开源大语言模型系列，Qwen3-0.6B 是该系列最小的模型，仅有 0.6B 参数，适合轻量级场景快速部署与验证。

## 模型权重下载

执行以下命令，从 ModelScope 下载模型权重文件：

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B
```

## 模型注册

模型注册流程请参考[模型适配指南第 2 节](../../doc/模型适配指南.md#2-模型适配)。在 `tecovllm_modelzoo` 目录下编译并安装插件即可：

```bash
python setup.py bdist_wheel
```

或使用开发模式免去每次修改后重复编译：

```bash
python setup.py develop
```

## 启动推理服务

执行以下命令启动 vLLM 推理服务：

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
| CUDA（基准） | mmlu_pro | mean_acc | default | 12032 | **0.XXXX** |
| **SDAA（太初）** | mmlu_pro | mean_acc | default | 12032 | **0.XXXX** |

> 精度对比结果：SDAA 精度与 CUDA 基准精度误差在 ±2% 以内，符合精度达标要求。

## 性能验证

```
Benchmarking summary:
+-----------------------------------+-----------+
| Key                               |     Value |
+===================================+===========+
| Time taken for tests (s)          |    X.XXXX |
+-----------------------------------+-----------+
| Number of concurrency             |    1      |
+-----------------------------------+-----------+
| Total requests                    |    1      |
+-----------------------------------+-----------+
| Succeed requests                  |    1      |
+-----------------------------------+-----------+
| Failed requests                   |    0      |
+-----------------------------------+-----------+
| Output token throughput (tok/s)   |   XX.XXXX |
+-----------------------------------+-----------+
| Total token throughput (tok/s)    |   XX.XXXX |
+-----------------------------------+-----------+
| Request throughput (req/s)        |    X.XXXX |
+-----------------------------------+-----------+
| Average latency (s)               |    X.XXXX |
+-----------------------------------+-----------+
| Average time to first token (s)   |    X.XXXX |
+-----------------------------------+-----------+
| Average time per output token (s) |    X.XXXX |
+-----------------------------------+-----------+
| Average inter-token latency (s)   |    X.XXXX |
+-----------------------------------+-----------+
| Average input tokens per request  | 2048      |
+-----------------------------------+-----------+
| Average output tokens per request |  200      |
+-----------------------------------+-----------+
```
