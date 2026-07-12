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
    """从单个性能 log 中提取 TTFT(ms)、TPOT(ms)，校验 Failed=0。"""
    content = Path(log_file).read_text(encoding="utf-8")

    # 定位 Benchmarking summary
    idx = content.find("Benchmarking summary:")
    if idx == -1:
        raise ValueError(f"未在 {log_file} 中找到 Benchmarking summary")

    section = content[idx:]
    ttft = tpot = None
    total = success = failed = None

    for line in section.split("\n"):
        line = line.strip()
        if not line:
            continue
        if ttft is None and "TTFT (ms)" in line:
            ttft = _extract_value(line)
        elif tpot is None and "TPOT (ms)" in line:
            tpot = _extract_value(line)
        elif "Total / Success / Failed" in line:
            total, success, failed = _extract_total_success_failed(line)
        if ttft is not None and tpot is not None and failed is not None:
            break

    if ttft is None:
        raise ValueError(f"未在 {log_file} 中找到 TTFT (ms)")
    if tpot is None:
        raise ValueError(f"未在 {log_file} 中找到 TPOT (ms)")
    if failed is None:
        raise ValueError(f"未在 {log_file} 中找到 Total / Success / Failed")

    if failed != 0:
        raise ValueError(f"{log_file} 存在 {failed} 条失败请求")
    if total != success:
        raise ValueError(f"{log_file} Total({total}) != Success({success})")

    return {"ttft_ms": ttft, "tpot_ms": tpot}


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
# 辅助
# ============================================================

def _extract_value(line: str) -> float:
    """从 │ Label │ 123.45 │ 中提取数值"""
    cols = [c.strip() for c in line.split("│")]
    # 取最后一个非空纯数字列
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
        print("用法: python result_analyse.py <log_dir> [prec|perf]", file=sys.stderr)
        sys.exit(1)

    log_dir = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "prec"

    if mode == "prec":
        results = parse_precision_dir(log_dir)
    elif mode == "perf":
        results = parse_performance_dir(log_dir)
    else:
        print(f"未知模式: {mode}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(results, indent=2, ensure_ascii=False))
