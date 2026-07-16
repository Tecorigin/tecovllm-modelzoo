#!/usr/bin/env python3
"""Epigenetic_Marks_Prediction-Histone completion eval."""
import argparse, json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
from tqdm import tqdm

DS_NAME = "Epigenetic_Marks_Prediction-Histone"
MS_REPO = "ZhejiangLab/RiceBenchmark"
MS_BASE = "https://www.modelscope.cn/datasets"


# ---- Data ----
def ensure_dataset(data_dir: str) -> str:
    """Return path to test.jsonl, downloading from ModelScope if missing."""
    ds_dir = os.path.join(data_dir, DS_NAME)
    test_file = os.path.join(ds_dir, "test.jsonl")
    if not os.path.exists(test_file):
        os.makedirs(ds_dir, exist_ok=True)
        url = f"{MS_BASE}/{MS_REPO}/resolve/master/{DS_NAME}/test.jsonl"
        print(f"Downloading {url} …")
        r = subprocess.run(["wget", "-q", "--show-progress", "-O", test_file, url], timeout=120)
        if r.returncode != 0:
            raise RuntimeError("Download failed")
    return test_file


def load_sequences(path: str, max_samples: int) -> list[str]:
    with open(path) as f:
        return [json.loads(l)["sequence"] for l in f if l.strip()][:max_samples]


# ---- Model ----
def fetch_model(host: str, port: int) -> str:
    return requests.get(f"http://{host}:{port}/v1/models", timeout=10).json()["data"][0]["id"]


# ---- Completion ----
def complete(prompt: str, api_url: str, model: str, n_bases: int,
             temperature: float = 0, timeout: int = 300) -> str | None:
    """Generate `n_bases` of continuation from vLLM.  Returns cleaned DNA string."""
    payload = {"model": model, "prompt": prompt,
               "max_tokens": n_bases * 2 + 10, "temperature": temperature}
    for _ in range(3):
        try:
            raw = requests.post(api_url, json=payload, timeout=timeout).json()
            clean = "".join(raw["choices"][0]["text"].split())
            return "".join(c for c in clean if c in "ACGTacgt").upper()[:n_bases]
        except Exception:
            time.sleep(1)
    return None


def run_completion(sequences: list[str], api_url: str, model: str,
                   n_bases: int, workers: int = 10) -> list[tuple[str | None, str]]:
    """
    Predict last `n_bases` for every sequence.  Returns list of (generated, truth).
    """
    items = [(s[:-n_bases], s[-n_bases:]) for s in sequences]
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {}
        for i, (prompt, truth) in enumerate(items):
            fut_map[ex.submit(complete, prompt, api_url, model, n_bases)] = (i, truth)
        for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="Generating", unit="seq"):
            i, truth = fut_map[fut]
            results[i] = (fut.result(), truth)

    return results


# ---- Metrics ----
def compute_metrics(results: list[tuple[str | None, str]], n_bases: int) -> dict:
    """Compute accuracy from (generated, truth) pairs."""
    per, pc, pt = [], np.zeros(n_bases), np.zeros(n_bases)
    for gen, truth in results:
        if gen is None:
            per.append(0); continue
        n = min(len(gen), n_bases)
        c = sum(1 for a, b in zip(gen[:n], truth[:n]) if a == b)
        per.append(c / n)
        for i in range(n):
            if gen[i] == truth[i]:
                pc[i] += 1
            pt[i] += 1

    pa = np.divide(pc, pt, where=pt > 0, out=np.zeros(n_bases, dtype=float))
    acc = np.array(per)
    failed = sum(1 for g, _ in results if g is None)

    return {
        "dataset": DS_NAME,
        "n_bases": n_bases,
        "n_samples": len(results),
        "n_failed": failed,
        "overall_accuracy": float(pc.sum() / pt.sum()) if pt.sum() else 0.0,
        "mean_accuracy": float(acc.mean()),
        "median_accuracy": float(np.median(acc)),
        "std_accuracy": float(acc.std()),
        "min_accuracy": float(acc.min()),
        "max_accuracy": float(acc.max()),
        "per_sample": acc.tolist(),
        "position_accuracy": pa.tolist(),
    }


def print_metrics(m: dict):
    print(f"\n{'='*50}")
    print(f"  {m['dataset']}  (last {m['n_bases']}bp)")
    print(f"{'='*50}")
    print(f"  Overall acc  : {m['overall_accuracy']:.4f}")
    print(f"  Mean / Med   : {m['mean_accuracy']:.4f} / {m['median_accuracy']:.4f}")
    print(f"  Std / Range  : {m['std_accuracy']:.4f} / [{m['min_accuracy']:.4f}, {m['max_accuracy']:.4f}]")
    print(f"  Failed       : {m['n_failed']}")
    K2 = m['n_bases'] // 2
    pa = m['position_accuracy']
    print(f"  Pos acc (first 10) : {[f'{x:.2f}' for x in pa[:10]]}")
    print(f"  Pos acc (mid   10) : {[f'{x:.2f}' for x in pa[K2-5:K2+5]]}")
    print(f"  Pos acc (last  10) : {[f'{x:.2f}' for x in pa[-10:]]}")
    print(f"{'='*50}")


# ---- CI Entry Point ----
def run_eval(host: str, port: int, data_dir: str = "./RiceBenchmark",
             max_samples: int = 10000, predict_bases: int = 100, workers: int = 256,
             output_dir: str | None = None) -> dict:
    """整套精度评测流程，供 CI (run_ci.py) 直接调用。

    Returns:
        {"epi_eval": overall_accuracy, "metrics": {...}, "output_file": str}
    """
    # 1. 数据
    test_file = ensure_dataset(data_dir)
    seqs = load_sequences(test_file, max_samples)

    # 2. 模型
    model = fetch_model(host, port)

    # 3. 运行
    api_url = f"http://{host}:{port}/v1/completions"
    print(f"\n[epi_eval] Server: {host}:{port}  |  Model: {model}")
    print(f"[epi_eval] Dataset: {DS_NAME}  |  Samples: {len(seqs)}  |  Predict: last {predict_bases}bp  |  Workers: {workers}")

    results = run_completion(seqs, api_url, model, predict_bases, workers)

    # 4. 指标
    metrics = compute_metrics(results, predict_bases)
    print_metrics(metrics)

    # 5. 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join(os.path.dirname(__file__) or ".",
                                  "prec_logs",
                                  f"OneGenomeRice_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)

    out_file = os.path.join(output_dir, "epi_eval_results.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved -> {out_file}")

    return {
        "RiceBenchmark": metrics["overall_accuracy"],
        "metrics": metrics,
        "output_file": out_file,
    }


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(description="RiceBenchmark completion eval")
    parser.add_argument("--host", "-H", default="localhost")
    parser.add_argument("--port", "-P", type=int, default=8000)
    parser.add_argument("--predict-bases", "-p", type=int, default=100)
    parser.add_argument("--max-samples", "-n", type=int, default=50)
    parser.add_argument("--workers", "-w", type=int, default=10)
    parser.add_argument("--data-dir", "-D", default="/nvmedata/application/juzh/RiceBenchmark")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    run_eval(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        predict_bases=args.predict_bases,
        workers=args.workers,
        output_dir=os.path.dirname(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
