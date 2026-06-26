# Seed-OSS-36B-Instruct

## 1. 模型概述

Seed-OSS 是由字节跳动种子团队开发的开源大型语言模型系列，具备强大的长上下文理解、推理、智能代理及通用能力，并提供丰富的开发者友好功能。尽管仅使用 12T  tokens 进行训练，Seed-OSS 在多个主流公开基准测试中表现优异。

---

## 2. 快速开始

### 2.1 模型权重下载

执行以下命令，从 ModelScope 下载模型权重文件：

```bash
modelscope download --model unsloth/Seed-OSS-36B-Instruct --local_dir ./Seed-OSS-36B-Instruct
```

### 2.2 模型注册

**步骤 1：安装 TecovLLM 和 vLLM 安装包**（如已安装可跳过）

```bash
pip install ./TecovLLM/vllm_whl/vllm.whl
pip install ./TecovLLM/vllm_whl/tecovllm.whl
```

**步骤 2：定位 TecovLLM 安装路径**

执行以下命令，返回结果中的 `Location` 字段即为包根目录，记为 `{vllm-root}`：

```bash
pip show vllm
```

**步骤 3：放置模型文件**

将模型组网文件 `seed_oss.py` 放到 `{vllm-root}/vllm_sdaa/models/` 目录下。

**步骤 4：注册模型到 TecovLLM**

在 `{vllm-root}/vllm_sdaa/models/__init__.py` 文件中添加注册代码：

```python
def register_model():
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "SeedOssForCausalLM", "vllm_sdaa.models.seed_oss:SeedOssForCausalLM"
    )
```

### 2.3 启动推理服务

执行以下命令启动 vLLM 推理服务：

```bash
vllm serve Seed-OSS-36B-Instruct \
    --served-model-name Seed-OSS-36B-Instruct \
    --tensor_parallel_size 8 \
    --port 8080 \
    --max_num_seqs 32 \
    --max_model_len 32768
```

### 2.4 精度验证

#### 2.4.1 运行精度测试

执行以下命令进行数据集精度验证：

```bash
evalscope eval \
    --model Seed-OSS-36B-Instruct \
    --api-url http://localhost:8080/v1/chat/completions \
    --api-key EMPTY_TOKEN \
    --datasets aime24 \
    --eval-batch-size 32 \
    --ignore-errors \
    --generation-config do_sample=true,temperature=1.0,top_p=0.95,top_k=40
```

#### 2.4.2 精度结果

| 平台 | 数据集 | 指标 | Subset | Num | Score |
|------|--------|------|--------|-----|-------|
| CUDA（基准） | aime24 | mean_acc | default | 30 | **0.7667** |
| **SDAA（太初）** | aime24 | mean_acc | default | 30 | **0.7667** |

> 精度对比结果：SDAA 精度与 CUDA 基准精度误差在 ±2% 以内，符合精度达标要求。

### 2.5 性能验证

#### 2.5.1 运行性能测试

执行以下命令进行模型推理性能验证：

```bash
evalscope perf \
  --parallel 1 \
  --model Seed-OSS-36B-Instruct \
  --url http://0.0.0.0:8080/v1/chat/completions \
  --api openai \
  --dataset random \
  --min-tokens 0 \
  --max-tokens 200 \
  --min-prompt-length 2048 \
  --max-prompt-length 2048 \
  --number 1 \
  --tokenizer-path /tecogpfs/models/ByteDance-Seed/Seed-OSS-36B-Instruct \
  --extra-args '{"ignore_eos": true}'
```

#### 2.5.2 性能结果

```
Benchmarking summary:
+-----------------------------------+-----------+
| Key                               |     Value |
+===================================+===========+
| Time taken for tests (s)          |    9.6167 |
+-----------------------------------+-----------+
| Number of concurrency             |    1      |
+-----------------------------------+-----------+
| Request rate (req/s)              |   -1      |
+-----------------------------------+-----------+
| Total requests                    |    1      |
+-----------------------------------+-----------+
| Succeed requests                  |    1      |
+-----------------------------------+-----------+
| Failed requests                   |    0      |
+-----------------------------------+-----------+
| Output token throughput (tok/s)   |   20.7972 |
+-----------------------------------+-----------+
| Total token throughput (tok/s)    |  233.761  |
+-----------------------------------+-----------+
| Request throughput (req/s)        |    0.104  |
+-----------------------------------+-----------+
| Average latency (s)               |    9.6167 |
+-----------------------------------+-----------+
| Average time to first token (s)   |    0.9584 |
+-----------------------------------+-----------+
| Average time per output token (s) |    0.0435 |
+-----------------------------------+-----------+
| Average inter-token latency (s)   |    0.0433 |
+-----------------------------------+-----------+
| Average input tokens per request  | 2048      |
+-----------------------------------+-----------+
| Average output tokens per request |  200      |
+-----------------------------------+-----------+
2026-05-19 10:48:58 - evalscope - INFO: 
Percentile results:
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
| Percentiles | TTFT (s) | ITL (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Output (tok/s) | Total (tok/s) |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
|     10%     |  0.9584  |  0.043  |  0.0435  |   9.6167    |     2048     |      200      |    20.7972     |   233.7611    |
|     25%     |  0.9584  | 0.0432  |  0.0435  |   9.6167    |     2048     |      200      |    20.7972     |   233.7611    |
|     50%     |  0.9584  | 0.0433  |  0.0435  |   9.6167    |     2048     |      200      |    20.7972     |   233.7611    |
|     66%     |  0.9584  | 0.0434  |  0.0435  |   9.6167    |     2048     |      200      |    20.7972     |   233.7611    |
|     75%     |  0.9584  | 0.0435  |  0.0435  |   9.6167    |     2048     |      200      |    20.7972     |   233.7611    |
|     80%     |  0.9584  | 0.0435  |  0.0435  |   9.6167    |     2048     |      200      |    20.7972     |   233.7611    |
|     90%     |  0.9584  | 0.0438  |  0.0435  |   9.6167    |     2048     |      200      |    20.7972     |   233.7611    |
|     95%     |  0.9584  | 0.0455  |  0.0435  |   9.6167    |     2048     |      200      |    20.7972     |   233.7611    |
|     98%     |  0.9584  | 0.0476  |  0.0435  |   9.6167    |     2048     |      200      |    20.7972     |   233.7611    |
|     99%     |  0.9584  | 0.0479  |  0.0435  |   9.6167    |     2048     |      200      |    20.7972     |   233.7611    |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
```
