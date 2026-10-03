#!/usr/bin/env python3
"""Hy-MT2-1.8B 融合 RMSNorm 正确性单元测试。

对比三方：
  - CPU fp32 独立参考（语义锚点）
  - transformers 原生 HunYuanDenseV1RMSNorm（bf16 baseline）
  - custom_ops 融合实现（bf16 -> fp16 -> tecoops.rms_norm -> bf16）

覆盖隐藏维 2048 / 512 / 128 / 64 与真实 QK-Norm 4D shape，以及 fp16 直通。
运行（须先 source setvars）：
  /home/py312/bin/python custom_ops/hy_mt2/test_rmsnorm_fused.py --device sdaa:0
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def cpu_ref(x, weight, eps):
    x32 = x.detach().float().cpu()
    w32 = weight.detach().float().cpu()
    var = x32.pow(2).mean(-1, keepdim=True)
    return (w32 * x32 * torch.rsqrt(var + eps)).to(x.dtype)


def err(a, b):
    b = b.float().to(a.device)
    d = (a.float() - b).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="sdaa:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch_sdaa  # noqa: F401
    from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
        HunYuanDenseV1RMSNorm,
    )
    from custom_ops.hy_mt2 import apply_rmsnorm_overlay, fused_rms_norm

    dev = args.device
    torch.manual_seed(args.seed)
    eps = 1e-5
    shapes = [(1, 1, 2048), (1, 30, 2048), (1, 1, 512), (4, 8),
              (1, 16, 1, 128), (1, 4, 1, 128), (1, 16, 30, 128)]
    results = {}
    failures = []

    for shape in shapes:
        h = shape[-1]
        x = torch.randn(*shape, dtype=torch.bfloat16, device=dev)
        w = torch.randn(h, dtype=torch.bfloat16, device=dev) * 0.5 + 1.0
        y_ref = cpu_ref(x, w, eps)

        # baseline（原生模块）
        m = HunYuanDenseV1RMSNorm(h, eps=eps).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            m.weight.copy_(w)
        with torch.inference_mode():
            y_base = m(x)
            y_fused = fused_rms_norm(x, w, eps)

        # overlay 绑定路径
        holder = torch.nn.Module()
        holder.norm = m
        bound = apply_rmsnorm_overlay(holder, verbose=False)
        with torch.inference_mode():
            y_ovl = holder.norm(x)

        rec = {
            "shape": list(shape),
            "bound_modules": bound,
            "fused_vs_cpu_fp32": err(y_fused, y_ref),
            "fused_vs_baseline_bf16": err(y_fused, y_base),
            "overlay_vs_fused": err(y_ovl, y_fused),
            "overlay_bitwise_eq_fused": bool(torch.equal(y_ovl, y_fused)),
        }
        results[str(shape)] = rec
        ok = (rec["fused_vs_cpu_fp32"]["max_abs"] < 5e-2
              and rec["fused_vs_baseline_bf16"]["max_abs"] < 5e-2
              and rec["overlay_bitwise_eq_fused"])
        if not ok:
            failures.append(str(shape))
        print(f"[{'PASS' if ok else 'FAIL'}] {shape} {json.dumps(rec['fused_vs_cpu_fp32'])} "
              f"vs_base={rec['fused_vs_baseline_bf16']['max_abs']:.3e} "
              f"overlay_eq={rec['overlay_bitwise_eq_fused']}", flush=True)

    # fp16 直通
    x16 = torch.randn(1, 8, 2048, dtype=torch.float16, device=dev)
    w16 = torch.randn(2048, dtype=torch.float16, device=dev) * 0.5 + 1.0
    with torch.inference_mode():
        y16 = fused_rms_norm(x16, w16, eps)
    e16 = err(y16, cpu_ref(x16, w16, eps))
    ok16 = e16["max_abs"] < 1e-2
    print(f"[{'PASS' if ok16 else 'FAIL'}] fp16-direct {json.dumps(e16)}", flush=True)
    results["fp16_direct"] = e16
    if not ok16:
        failures.append("fp16_direct")

    # fp32 必须显式拒绝
    try:
        fused_rms_norm(torch.randn(1, 8, 64, dtype=torch.float32, device=dev),
                       torch.randn(64, dtype=torch.float32, device=dev), eps)
        print("[FAIL] fp32 should raise ValueError", flush=True)
        failures.append("fp32_guard")
    except ValueError:
        print("[PASS] fp32 raises ValueError", flush=True)

    print("SUMMARY", json.dumps({"failures": failures, "results": results}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
