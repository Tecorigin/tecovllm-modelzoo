#!/usr/bin/env python3
"""run_ci.py — CI 自动化流程：起服务 → 精度 → 性能 → 提取结果

用法: python run_ci.py <run.sh路径> [--keep-service]
  run.sh: 启动 vLLM 服务的脚本，需包含 --served-model-name 和 --port
"""

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from result_analyse import parse_precision_dir, parse_performance_dir

SCRIPT_DIR = Path(__file__).resolve().parent


SUPPORTED_MODELS = {"Hy-MT2-1.8B", "MiniCPM5-1B", "InternVL3_5-8B", "gemma-4-12B-it"}


def parse_run_sh(path: str) -> dict:
    """从 run.sh 解析 --served-model-name、模型路径、--port、--host（跳过注释行）"""
    lines = Path(path).read_text().split("\n")
    # 去除注释行（# 开头的行）
    active = "\n".join(
        l for l in lines if not l.strip().startswith("#")
    )
    content = active

    model_name = None
    m = re.search(r"--served-model-name\s+(\S+)", content)
    if m:
        model_name = m.group(1)
    if not model_name or model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"--served-model-name 必须为 {SUPPORTED_MODELS}，实际: {model_name}"
        )

    if "--no-enable-prefix-caching" not in active:
        raise ValueError("run.sh 中必须包含 --no-enable-prefix-caching")

    model_path = None
    m = re.search(r"vllm\s+serve\s+(\S+)", content)
    if m:
        model_path = m.group(1)

    port = 8000
    m = re.search(r"--port\s+(\d+)", content)
    if m:
        port = int(m.group(1))

    host = "0.0.0.0"
    m = re.search(r"--host\s+(\S+)", content)
    if m:
        host = m.group(1)

    return {"model_name": model_name, "model_path": model_path, "port": port, "host": host}


def wait_service(host: str, port: int, timeout: int = 600) -> bool:
    """轮询 /health 直到 200，超时抛异常"""
    url = f"http://{host}:{port}/health"
    print(f"等待服务就绪: {url}  (最长 {timeout}s)")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                elapsed = timeout - (deadline - time.time())
                print(f"服务已就绪 ({elapsed:.0f}s)")
                return True
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"服务启动超时 ({timeout}s): {url}")


def run_cmd(args, cwd=None):
    """运行命令，实时输出，失败直接抛异常"""
    print(f"\n>>> {' '.join(args)}")
    p = subprocess.run(args, cwd=cwd or SCRIPT_DIR)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败 (exit={p.returncode}): {' '.join(args)}")


def stop_service(proc):
    """停止 vLLM 进程及其所有子进程"""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        proc.wait()
    print("vLLM 服务已停止")


def find_latest_log(pattern: str) -> Path | None:
    """找最新匹配的日志目录"""
    dirs = sorted(SCRIPT_DIR.glob(pattern))
    return dirs[-1] if dirs else None


def compare_precision(model_name: str, results: dict) -> None:
    """对比精度结果与基线，不达标直接抛异常"""
    baseline_path = SCRIPT_DIR / "precision_baseline.json"
    if not baseline_path.exists():
        print("未找到 precision_baseline.json，跳过精度对比")
        return

    baseline_all = json.loads(baseline_path.read_text())
    model_baseline = baseline_all.get(model_name)
    if not model_baseline:
        print(f"基线中无 {model_name}，跳过精度对比")
        return

    print(f"\n精度对比 ({model_name}):")
    all_pass = True
    for dataset, expected in model_baseline.items():
        actual = results.get(dataset)
        if actual is None:
            print(f"  {dataset}: 缺失 (需要 >= {expected['min']})  FAIL")
            all_pass = False
            continue
        lo, hi = expected["min"], expected["max"]
        ok = lo <= actual <= hi
        status = "PASS" if ok else "FAIL"
        print(f"  {dataset}: actual={actual:.4f}  baseline={expected['score']:.4f}  range=[{lo:.4f}, {hi}]  {status}")
        if not ok:
            all_pass = False
    if not all_pass:
        raise ValueError(f"模型 {model_name} 精度不达标，见上文")


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CI 自动化：起服务 → 精度 → 性能 → 提取结果")
    parser.add_argument("run_sh", help="启动 vLLM 服务的脚本路径（如 model_adaptations/team1/run.sh）")
    parser.add_argument("--keep-service", action="store_true", help="测试完成后保留 vLLM 服务不关闭")
    args = parser.parse_args()

    run_sh_path = args.run_sh
    keep_service = args.keep_service

    info = parse_run_sh(run_sh_path)

    model_name = info["model_name"]
    model_path = info["model_path"]
    port = info["port"]
    host = info["host"]

    if not model_path:
        print("错误: run.sh 中未找到 vllm serve <模型路径>", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"  模型:  {model_name}")
    print(f"  路径:  {model_path}")
    print(f"  地址:  {host}:{port}")
    print(f"  脚本:  {run_sh_path}")
    print("=" * 60)

    # ---- 清理旧日志 ----
    for d in ["outputs", "prec_logs", "speed_logs"]:
        p = SCRIPT_DIR / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"已清理: {d}")

    # ---- Step 1: 启动 vLLM ----
    print("\n[1/5] 启动 vLLM 服务 ...")
    log_file = SCRIPT_DIR / f"vllm_serve.log"
    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(
            ["bash", run_sh_path],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(Path(run_sh_path).parent),
            start_new_session=True,
        )

    try:
        wait_service(host, port)

        # ---- Step 2: 精度测试 ----
        print("\n[2/5] 精度测试 ...")
        run_cmd([
            "bash", str(SCRIPT_DIR / "prec.sh"),
            model_name, host, str(port),
        ])

        # ---- Step 2.5: 精度对比 ----
        prec_dir = find_latest_log(f"prec_logs/{model_name}_*")
        if prec_dir:
            results = parse_precision_dir(str(prec_dir))
            out = json.dumps(results, ensure_ascii=False, indent=2)
            print(out)
            (SCRIPT_DIR / "precision_result.json").write_text(out, encoding="utf-8")
            compare_precision(model_name, results)
        else:
            raise RuntimeError("未找到精度日志目录")

        # ---- Step 3: 性能测试 ----
        print("\n[3/5] 性能测试 ...")
        run_cmd([
            "bash", str(SCRIPT_DIR / "speed.sh"),
            model_name, model_path, host, str(port),
        ])

    except Exception:
        if not keep_service:
            print("\n!!! 测试失败，停止 vLLM 服务 ...")
            stop_service(proc)
        raise

    # ---- Step 4: 停止服务 ----
    if not keep_service:
        print("\n[4/5] 停止 vLLM 服务 ...")
        stop_service(proc)
    else:
        print("\n[4/5] 保留 vLLM 服务 (--keep-service)")

    # ---- Step 5: 提取结果 ----
    print("\n[5/5] 提取结果 ...")

    speed_dir = find_latest_log(f"speed_logs/{model_name}_*")
    if speed_dir:
        results = parse_performance_dir(str(speed_dir))
        out = json.dumps(results, ensure_ascii=False, indent=2)
        print(out)
        (SCRIPT_DIR / "performance_result.json").write_text(out, encoding="utf-8")
    else:
        raise RuntimeError("未找到性能日志目录")

    print("\n===== CI 完成 =====")


if __name__ == "__main__":
    main()
