#!/usr/bin/env python3
"""Hy-MT2-1.8B 融合 SwiGLU MLP 正确性单元测试。

对比四方：
  - CPU fp32 独立参考（语义锚点）
  - transformers 原生 HunYuanDenseV1MLP forward（bf16 baseline）
  - custom_ops 参考融合实现 fused_mlp（合并 GEMM + 融合 swiglu）
  - overlay 绑定路径（实例 forward 替换）

覆盖真实 MLP shape (2048->6144) 与若干小 shape；含 fp32 跳过与幂等/移除检查。
运行（须先 source setvars）：
  /home/py312/bin/python custom_ops/hy_mt2/test_swiglu_fused.py --device sdaa:0
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def cpu_ref_mlp(x, wg, wu, wd):
    x32 = x.detach().float().cpu()
    g = F.linear(x32, wg.detach().float().cpu())
    u = F.linear(x32, wu.detach().float().cpu())
    act = F.silu(g) * u
    return F.linear(act, wd.detach().float().cpu())


def err(a, b):
    b = b.float().to(a.device)
    d = (a.float() - b).abs()
    denom = b.abs().max().clamp_min(1e-6)
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rel_max": (d.max() / denom).item(),
    }


def make_mlp(hidden, inter, device, dtype, layer_idx=0):
    from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
        HunYuanDenseV1MLP,
    )

    cfg = SimpleNamespace(hidden_size=hidden, intermediate_size=inter, hidden_act="silu")
    m = HunYuanDenseV1MLP(cfg, layer_idx=layer_idx).to(device)
    m.to(dtype)
    with torch.no_grad():
        m.gate_proj.weight.copy_(torch.randn(inter, hidden, dtype=dtype, device=device) * 0.02)
        m.up_proj.weight.copy_(torch.randn(inter, hidden, dtype=dtype, device=device) * 0.02)
        m.down_proj.weight.copy_(torch.randn(hidden, inter, dtype=dtype, device=device) * 0.02)
    m.eval()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="sdaa:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch_sdaa  # noqa: F401
    from custom_ops.hy_mt2 import (
        apply_swiglu_overlay,
        fused_mlp,
        fused_swiglu,
        remove_swiglu_overlay,
    )

    dev = args.device
    torch.manual_seed(args.seed)
    results = {}
    failures = []

    # hidden/intermediate 对（intermediate 使 2I 为 16 的倍数）
    shapes = [(2048, 6144), (512, 1024), (256, 512), (64, 128)]
    tokens = [1, 30]

    for hidden, inter in shapes:
        for T in tokens:
            x = torch.randn(1, T, hidden, dtype=torch.bfloat16, device=dev)
            m = make_mlp(hidden, inter, dev, torch.bfloat16)

            with torch.inference_mode():
                y_base = m(x)
                w_gu = torch.cat(
                    [m.gate_proj.weight.detach(), m.up_proj.weight.detach()], dim=0
                ).contiguous()
                y_fused = fused_mlp(x, w_gu, m.down_proj.weight.detach(), inter)
                y_cpu = cpu_ref_mlp(x, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)

            holder = torch.nn.Module()
            holder.mlp = m
            bound = apply_swiglu_overlay(holder, verbose=False)
            with torch.inference_mode():
                y_ovl = holder.mlp(x)
            removed = remove_swiglu_overlay(holder)
            with torch.inference_mode():
                y_restored = holder.mlp(x)

            rec = {
                "hidden": hidden,
                "intermediate": inter,
                "tokens": T,
                "bound_modules": bound,
                "fused_vs_cpu_fp32": err(y_fused, y_cpu),
                "fused_vs_baseline_bf16": err(y_fused, y_base),
                "overlay_vs_fused": err(y_ovl, y_fused),
                "overlay_bitwise_eq_fused": bool(torch.equal(y_ovl, y_fused)),
                "restored_bitwise_eq_baseline": bool(torch.equal(y_restored, y_base)),
                "removed_modules": removed,
            }
            results[f"{hidden}x{inter}_T{T}"] = rec
            ok = (
                bound == 1
                and removed == 1
                and rec["overlay_bitwise_eq_fused"]
                and rec["restored_bitwise_eq_baseline"]
                and rec["fused_vs_cpu_fp32"]["rel_max"] < 5e-2
                and rec["fused_vs_baseline_bf16"]["rel_max"] < 5e-2
            )
            if not ok:
                failures.append(f"{hidden}x{inter}_T{T}")
            print(
                f"[{'PASS' if ok else 'FAIL'}] {hidden}x{inter} T={T} "
                f"vs_cpu_rel={rec['fused_vs_cpu_fp32']['rel_max']:.2e} "
                f"vs_base_rel={rec['fused_vs_baseline_bf16']['rel_max']:.2e} "
                f"ovl_eq={rec['overlay_bitwise_eq_fused']} restored_eq={rec['restored_bitwise_eq_baseline']}",
                flush=True,
            )

    # 融合 swiglu 语义锚点：fp32 精确、fp16/bf16 精度量级
    for dt, tol in ((torch.float32, 0.0), (torch.float16, 2e-3), (torch.bfloat16, 2e-2)):
        gu = torch.randn(1, 8, 1, 2 * 128, dtype=dt, device=dev)
        a, b = gu.chunk(2, dim=-1)
        ref = F.silu(a.float()) * b.float()
        with torch.inference_mode():
            out = fused_swiglu(gu, 128)
        e = err(out, ref)
        ok = e["rel_max"] <= max(tol, 1e-9) if dt != torch.float32 else e["max_abs"] == 0.0
        print(f"[{'PASS' if ok else 'FAIL'}] swiglu {dt} {json.dumps(e)}", flush=True)
        results[f"swiglu_{dt}"] = e
        if not ok:
            failures.append(f"swiglu_{dt}")

    # fp32 模块必须被跳过（不静默绑定）
    holder = torch.nn.Module()
    holder.bf16 = make_mlp(64, 128, dev, torch.bfloat16, layer_idx=0)
    holder.fp32 = make_mlp(64, 128, dev, torch.float32, layer_idx=1)
    bound = apply_swiglu_overlay(holder, verbose=False)
    ok_skip = bound == 1 and "_hy_fused_swiglu" not in holder.fp32.__dict__
    print(f"[{'PASS' if ok_skip else 'FAIL'}] fp32 skipped (bound={bound})", flush=True)
    if not ok_skip:
        failures.append("fp32_skip")

    print("SUMMARY", json.dumps({"failures": failures, "results": results}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
