#!/usr/bin/env python3
"""run_ci.py — CI 自动化流程：起服务 → 精度 → 性能 → 提取结果

用法: python run_ci.py <run.sh路径> [--keep-service]
  run.sh: 启动 vLLM 服务的脚本，必须包含 --no-enable-prefix-caching 和 --trust-remote-code
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

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import epi_eval

SUPPORTED_MODELS = {"Intern-S2-Preview", "OneGenomeRice"}


def parse_run_sh(path: str) -> dict:
    lines = Path(path).read_text().split("\n")
    active = "\n".join(l for l in lines if not l.strip().startswith("#"))
    content = active

    # 校验必须包含的参数
    for flag in ["--no-enable-prefix-caching", "--trust-remote-code"]:
        if flag not in content:
            raise ValueError(f"run.sh 中必须包含 {flag} 参数")

    model_name = None
    m = re.search(r"--served-model-name\s+(\S+)", content)
    if m:
        model_name = m.group(1)
    if not model_name or model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"--served-model-name 必须为 {SUPPORTED_MODELS}，实际: {model_name}"
        )

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
    print(f"\n>>> {' '.join(args)}")
    p = subprocess.run(args, cwd=cwd or SCRIPT_DIR)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败 (exit={p.returncode}): {' '.join(args)}")


def stop_service(proc):
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
    dirs = sorted(SCRIPT_DIR.glob(pattern))
    return dirs[-1] if dirs else None


def compare_precision(model_name: str, results: dict) -> None:
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
    parser.add_argument("run_sh", help="启动 vLLM 服务的脚本路径")
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

    for d in ["outputs", "prec_logs", "speed_logs"]:
        p = SCRIPT_DIR / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"已清理: {d}")

    # ---- Step 1: 启动 vLLM ----
    print("\n[1/4] 启动 vLLM 服务 ...")
    log_file = SCRIPT_DIR / "vllm_serve.log"
    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(
            ["bash", run_sh_path],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    try:
        wait_service(host, port)
        print(f"服务日志: {log_file}")
    except Exception:
        stop_service(proc)
        raise

    # ---- Step 2: 精度测试 ----
    print("\n[2/4] 精度测试 ...")
    if model_name == "OneGenomeRice":
        result = epi_eval.run_eval(host=host, port=port, data_dir="/nvmedata/application/juzh/RiceBenchmark")
        prec_results = {"RiceBenchmark": result["RiceBenchmark"]}
    else:
        prec_sh = SCRIPT_DIR / "prec.sh"
        run_cmd(["bash", str(prec_sh), model_name, host, str(port)])
        prec_log = find_latest_log(f"prec_logs/{model_name}_*")
        if prec_log:
            prec_results = parse_precision_dir(str(prec_log))
            print(f"精度结果: {json.dumps(prec_results, indent=2)}")
        else:
            prec_results = {}
            print("警告: 未找到精度日志目录")

    # ---- Step 3: 精度对比 ----
    if prec_results:
        compare_precision(model_name, prec_results)

    # ---- Step 4: 性能测试 ----
    print("\n[3/4] 性能测试 ...")
    speed_sh = SCRIPT_DIR / "speed.sh"
    run_cmd(["bash", str(speed_sh), model_name, model_path, host, str(port)])

    # ---- Step 5: 提取性能结果 ----
    speed_log = find_latest_log(f"speed_logs/{model_name}_*")
    if speed_log:
        speed_results = parse_performance_dir(str(speed_log))
        print(f"性能结果: {json.dumps(speed_results, indent=2)}")
    else:
        print("警告: 未找到性能日志目录")

    # ---- 停止服务 / 保留 ----
    if keep_service:
        print(f"\n[4/4] 保留 vLLM 服务运行中 ({host}:{port})，PID: {proc.pid}")
    else:
        print("\n[4/4] 停止 vLLM 服务 ...")
        stop_service(proc)

    print("\n===== CI 完成 =====")


if __name__ == "__main__":
    main()
