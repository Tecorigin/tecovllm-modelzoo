# MiniCPM5-1B (SCU都队)

## 模型概述

MiniCPM5-1B 是面壁智能（OpenBMB）推出的端侧轻量级大语言模型，参数量约为 1B，具备优秀的推理性能与多任务理解能力。本项目为中国教育技术协会 2026 年“AI+教育”创新应用技能大赛“AI+加速卡模型适配赛道”SCU都队提交的适配工程。

- **赛题序号**：4
- **难度系数**：0.5
- **权重位置**：`/gpfs/model/OpenBMB/MiniCPM5-1B`
- **适配框架**：TecoVLLM / vLLM-SDAA

## 权重说明

依据组委会纪律，所有评测严格基于服务器预置只读权重，禁止重复下载或转储：
`/gpfs/model/OpenBMB/MiniCPM5-1B`

## 启动推理服务

vLLM 推理服务启动脚本为 `run.sh`，符合官方 CI 合约：

```bash
MODEL_ROOT=/gpfs/model bash model_adaptations/MiniCPM5SCUdoudui/run.sh
```

启动前必须按官方流程安装 `teco-ops-SCUdoudui` 构建的 `tecoops` wheel。
`run.sh` 会从当前适配目录加载 `runtime/overlay/sitecustomize.py`，在进程初始化时
一次性绑定 `rms_norm`、`reshape_and_cache` 和 prefill
`flash_attn_varlen_func`；缺少任一官方接口时服务 fail-closed，不回退到未审计的
Python 热路径。

核心参数配置：
- `--served-model-name MiniCPM5-1B`
- `--tensor-parallel-size 1`
- `--port 8000`
- `--host 0.0.0.0`
- `--dtype float16`
- `--trust-remote-code`
- `--no-enable-prefix-caching`
- `--max-model-len 32768`

## 精度验证规范

依据官方赛道评测标准：
- **评测工具**：`evalscope`
- **数据集**：`mmlu_pro` (subset: `computer science`)
- **评测规模**：limit 200 cases
- **CUDA 基准分**：0.345
- **SDAA 达标区间**：[0.245, 0.445]（基线 ±0.1）

评测命令（通过 `tools/ci_pipline/prec.sh` 触发）：
```bash
./tools/ci_pipline/prec.sh MiniCPM5-1B 0.0.0.0 8000
```

## 性能验证规范

依据官方统一性能测试口径（通过 `tools/ci_pipline/speed.sh` 触发）：
- **预热 (Warmup)**：1024 输入 -> 10 输出，batch size 1
- **文本测试场景**：
  - T1: 2048 输入 -> 100 输出 (bs=1, number=10)
  - T2: 4096 输入 -> 100 输出 (bs=1, number=10)
  - T3: 8192 输入 -> 100 输出 (bs=1, number=10)
  - T4: 16384 输入 -> 100 输出 (bs=1, number=10)
- **核心指标**：TTFT (Time To First Token) 与 TPOT (Time Per Output Token)

## 重点算子优化与框架接入

模型在太初 SDAA 上接入并协同优化三个官方指定算子（提交至 `teco-ops-SCUdoudui`）：
1. `flash_attn_varlen_func`：Prefill 阶段变长注意力，针对 GQA（num_heads=16, num_kv_heads=2）与因果掩码优化；
2. `reshape_and_cache`：Decode 阶段 PagedAttention 分页 KV Cache 组装；
3. `rms_norm`：双缓冲流水与 Fused Residual Add 优化。

模型接入代码位于 `runtime/custom_ops/`；它只负责模型框架到官方 `tecoops` ABI
的映射，不复制算子 kernel。decode 单 token 继续使用厂商 BlockAttention，prefill
实际调用官方 flash ABI。可通过设置 `TECOOPS_CALL_RECEIPT=/tmp/tecoops_calls.log`
收集三接口首次真实调用凭证。

## CI 验证命令

```bash
python tools/ci_pipline/run_ci.py model_adaptations/MiniCPM5SCUdoudui/run.sh
```
