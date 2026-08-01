#!/usr/bin/env python3
"""result_analyse.py — 从 prec_logs / speed_logs 中提取精度和性能结果"""

import re
from pathlib import Path


# ============================================================
# 精度解析
# ============================================================

def parse_precision(log_file: str, metric: str = None) -> float:
    """从精度 log 提取 Score。metric 指定目标指标名(如 mean_bert_score)，None 时取首个数据行"""
    content = Path(log_file).read_text(encoding="utf-8")

    idx = content.find("Overall report table")
    if idx == -1:
        raise ValueError(f"未在 {log_file} 中找到 Overall report table")

    section = content[idx:]
    lines = [l.strip() for l in section.split("\n") if l.strip()]

    # 找表头行确认列位置
    score_col = -1
    for line in lines:
        if "Score" in line and "│" in line:
            cols = [c.strip() for c in line.split("│")]
            for i, c in enumerate(cols):
                if c == "Score":
                    score_col = i
            break

    if score_col < 0:
        raise ValueError(f"未在 {log_file} 中找到 Score 列")

    # 找 Metric 列位置
    metric_col = -1
    for line in lines:
        if "Metric" in line and "│" in line:
            cols = [c.strip() for c in line.split("│")]
            for i, c in enumerate(cols):
                if c == "Metric":
                    metric_col = i
            break

    # 找匹配的数据行
    for line in lines:
        if "│" not in line or "─" in line:
            continue
        cols = [c.strip() for c in line.split("│")]
        if score_col >= len(cols):
            continue
        if "Score" in line:
            continue
        if metric and metric_col >= 0:
            if metric_col < len(cols) and cols[metric_col] == metric:
                return float(cols[score_col])
        elif not metric:
            return float(cols[score_col])

    raise ValueError(f"未在 {log_file} 中找到指标 {metric or '(any)'}")


# 数据集 → 目标指标
DATASET_METRIC = {
    "wmt24pp": "mean_bert_score",
    "mmlu_pro": "mean_acc",
    "mmmu_pro": "mean_acc",
}


def parse_precision_dir(log_dir: str) -> dict:
    """遍历精度日志目录，返回 {数据集名: Score}"""
    results = {}
    for f in sorted(Path(log_dir).glob("*.log")):
        dataset = f.stem
        target_metric = DATASET_METRIC.get(dataset)
        results[dataset] = parse_precision(str(f), metric=target_metric)
    if not results:
        raise ValueError(f"{log_dir} 中无 log 文件")
    return results


# ============================================================
# 性能解析
# ============================================================

def parse_performance(log_file: str) -> dict:
    """从单个性能 log 的 Percentile results 表格中提取 TTFT (ms) 和 TPOT (ms) 的 1% 值。"""
    content = Path(log_file).read_text(encoding="utf-8")

    # 定位 EvalScope 百分位结果表格
    idx = content.rfind("Percentile results:")
    if idx == -1:
        raise ValueError(f"未在 {log_file} 中找到 Percentile results:")

    section = content[idx:]
    lines = section.split("\n")

    p1_col = -1
    metric_col = -1
    header_found = False

    ttft_p1 = None
    tpot_p1 = None

    for line in lines:
        # rich 表格的表头使用 ┃，数据行使用 │；二者列索引一致。
        if not header_found and ("┃" in line or "│" in line):
            cols = [c.strip() for c in re.split(r"[┃│]", line)]
            for i, c in enumerate(cols):
                if c == "1%":
                    p1_col = i
                elif c == "Metric":
                    metric_col = i
            header_found = p1_col >= 0 and metric_col >= 0
            continue

        # 解析数据行
        if header_found and "│" in line:
            cols = [c.strip() for c in re.split(r"[┃│]", line)]
            if metric_col < len(cols) and p1_col < len(cols):
                metric = cols[metric_col]
                if metric == "TTFT (ms)" and ttft_p1 is None:
                    ttft_p1 = float(cols[p1_col])
                elif metric == "TPOT (ms)" and tpot_p1 is None:
                    tpot_p1 = float(cols[p1_col])

    if ttft_p1 is None:
        raise ValueError(f"未在 {log_file} 的 Percentile results 中找到 TTFT (ms) 1%")
    if tpot_p1 is None:
        raise ValueError(f"未在 {log_file} 的 Percentile results 中找到 TPOT (ms) 1%")

    return {"ttft_ms": ttft_p1, "tpot_ms": tpot_p1}


def parse_performance_dir(log_dir: str) -> dict:
    """遍历性能日志目录，返回 {测试名: {ttft_ms, tpot_ms}}"""
    results = {}
    for f in sorted(Path(log_dir).glob("*.log")):
        if f.stem == "warmup":
            continue
        results[f.stem] = parse_performance(str(f))
    if not results:
        raise ValueError(f"{log_dir} 中无 log 文件")
    return results


# ============================================================
# 性能评分
# ============================================================

# 性能参考基准（B=最差基线, Best=最优参考, w=case权重）
# 来源: doc/模型适配指南.md §4.4
_PERFORMANCE_REF = {
    "OneGenomeRice": {
        "T1": {
            "ttft": {"B": 6400.00, "Best": 150},
            "tpot": {"B": 45.00, "Best": 31.00},
        },
        "T2": {
            "ttft": {"B": 21000.00, "Best": 390},
            "tpot": {"B": 65.00, "Best": 31.00},
        },
        "T3": {
            "ttft": {"B": 75000.00, "Best": 1100.00},
            "tpot": {"B": 110.00, "Best": 31.00},
        },
        "T4": {
            "ttft": {"B": 280000.00, "Best": 3600.00},
            "tpot": {"B": 200.00, "Best": 31.00},
        },
    }
}

_W = 0.25  # 每个 case 的权重
_MAX_PER_METRIC = _W * 25  # 单项指标最高得分 = 6.25


def calculate_performance_score(model_name: str, results: dict) -> dict:
    """根据公式计算性能得分。

    Score_i,M = (B_i,M - Actual_i,M) / (B_i,M - Best_i,M) × w_i × 25

    返回:
        {
            "cases": {
                "T1_in32768_out100": {
                    "ttft_ms": 3662.7, "tpot_ms": 23.9,
                    "ttft_score": 6.25, "tpot_score": 6.25,
                    "case_total": 12.50
                },
                ...
            },
            "total_score": 50.00,
            "max_score": 50.00
        }
    """
    ref = _PERFORMANCE_REF.get(model_name)
    if not ref:
        raise ValueError(f"未找到模型 {model_name} 的性能参考数据")

    total_score = 0.0
    enriched_cases = {}

    for case_key, metrics in results.items():
        case_id = case_key.split("_")[0]  # "T1_in32768_out100" -> "T1"
        case_ref = ref.get(case_id)
        if not case_ref:
            print(f"⚠️  警告: 未找到 {case_id} 的参考数据，跳过 {case_key}")
            enriched_cases[case_key] = {**metrics}
            continue

        case_data = {**metrics}  # 保留原始指标
        case_sum = 0.0

        for metric in ("ttft", "tpot"):
            key = f"{metric}_ms"
            actual = metrics.get(key)
            if actual is None:
                continue

            b = case_ref[metric]["B"]
            best = case_ref[metric]["Best"]

            score = (b - actual) / (b - best) * _MAX_PER_METRIC
            score = min(score, _MAX_PER_METRIC)  # 封顶
            score = max(score, 0)                 # 保底
            score = round(score, 4)

            case_data[f"{metric}_score"] = score
            case_sum += score

        case_sum = round(case_sum, 4)
        case_data["case_total"] = case_sum
        total_score += case_sum
        enriched_cases[case_key] = case_data

    total_score = round(total_score, 4)
    max_score = len(ref) * 2 * _MAX_PER_METRIC

    return {
        "cases": enriched_cases,
        "total_score": total_score,
        "max_score": max_score,
    }


# ============================================================
# 辅助
# ============================================================

def _extract_value(line: str) -> float:
    """从 │ Label │ 123.45 │ 中提取数值"""
    cols = [c.strip() for c in line.split("│")]
    for c in reversed(cols):
        try:
            return float(c)
        except ValueError:
            continue
    raise ValueError(f"无法从行中提取数值: {line}")


def _extract_total_success_failed(line: str):
    """从 │ Total / Success / Failed │ 20 / 20 / 0 │ 中提取三个数"""
    cols = [c.strip() for c in line.split("│")]
    for c in cols:
        m = re.match(r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", c)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
    raise ValueError(f"无法解析 Total/Success/Failed: {line}")


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("用法: python result_analyse.py <log_dir> [prec|perf|score] [model_name]", file=sys.stderr)
        sys.exit(1)

    log_dir = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "prec"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "OneGenomeRice"

    if mode == "prec":
        results = parse_precision_dir(log_dir)
    elif mode == "perf":
        results = parse_performance_dir(log_dir)
    elif mode == "score":
        perf_results = parse_performance_dir(log_dir)
        results = calculate_performance_score(model_name, perf_results)
    else:
        print(f"未知模式: {mode}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(results, indent=2, ensure_ascii=False))
