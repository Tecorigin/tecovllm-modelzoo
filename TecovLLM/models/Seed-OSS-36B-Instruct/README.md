# 模型名称

## 1. 模型概述

Seed-OSS 是由字节跳动种子团队开发的一系列开源大型语言模型，旨在提供强大的长上下文、推理、代理和通用能力以及多样化的开发者友好功能。尽管仅使用了 12T 令牌进行训练，Seed-OSS 在多个流行的开放基准测试中表现出色。

## 2. 快速开始
### 2.1 模型权重下载

执行以下命令，从modelscope下载模型权重文件。
```
modelscope download --model unsloth/Seed-OSS-36B-Instruct --local_dir ./Seed-OSS-36B-Instruct
```

### 2.2 模型注册

1. 安装TecovLLM和vLLM whl包(若环境中已安装则跳过该步骤)。
```
pip install ./TecovLLM/vllm_whl/vllm.whl
pip install ./TecovLLM/vllm_whl/tecovllm.whl
```
2. 定位TecovLLM包安装路径：在当前环境中执行查询命令`pip show vllm`，返回结果中的Location字段即为包所在根目录`{vllm-root}`。
3. 放置模型文件：将模型组网文件`seed_oss.py`放在`{vllm-root}/vllm_sdaa/models/`目录下。
4. 注册模型到TecovLLM。在`{vllm-root}/vllm_sdaa/models/init.py`文件中，添加模型注册代码，将步骤一实现的模型，注册到Teco-vLLM中。注册代码如下：
```
def register_model():
    # 导入模块注册模型模块
    from vllm import ModelRegistry
    # 注册模型
    ModelRegistry.register_model(
        "SeedOssForCausalLM", "vllm_sdaa.models.seed_oss:SeedOssForCausalLM")
```

### 2.3 启动推理

执行以下命令，启动vLLM推理服务。
```
vllm serve Seed-OSS-36B-Instruct --served-model-name Seed-OSS-36B-Instruct --tensor_parallel_size 8 --port 8080 --max_num_seqs 32 --max_model_len 32768
```

### 2.4 精度验证

执行以下命令，进行数据集精度验证。
```
evalscope eval \
    --model Seed-OSS-36B-Instruct \
    --api-url http://hocalhost:8080/v1/chat/completions \
    --api-key EMPTY_TOKEN \
    --datasets aime24 \
    --eval-batch-size 32 \
    --ignore-errors \
    --generation-config do_sample=true,temperature=1.0,top_p=0.95,top_k=40
```
精度测试结果如下：
```
+-----------------------+-----------+----------+----------+-------+---------+---------+
| Model                 | Dataset   | Metric   | Subset   |   Num |   Score | Cat.0   |
+=======================+===========+==========+==========+=======+=========+=========+
| Seed-OSS-36B-Instruct | aime24    | mean_acc | default  |    30 |  0.7667 | default |
+-----------------------+-----------+----------+----------+-------+---------+---------+ 
```

### 2.5 性能验证

执行以下命令，进行模型推理性能验证。
```
evalscope perf \
  --parallel 1 \
  --model Seed-OSS-36B-Instruct  \
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
性能测试结果如下：
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