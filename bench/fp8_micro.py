#!/usr/bin/env python3
"""FP8 GEMM microbenchmark at the Wan2.1-1.3B linear shapes (L4, sm_89).

Shapes (M tokens x K -> N): q/k/v/o 1536->1536, FFN up 1536->8960, FFN down 8960->1536, with
M = 14,040 (33 frames) and 32,760 (81 frames).

For each shape:
- bf16 F.linear (the baseline GEMM)
- FP8 GEMM only (_scaled_mm on pre-quantized inputs): tensorwise, rowwise, rowwise + fast_accum
- FP8 rowwise including activation quantization, compiled (what a block pays)
Times are CUDA-event means after warmup; errors are relative L2 against an fp32 matmul of the same
bf16 inputs. Two activation kinds: Gaussian, and Gaussian with 8 outlier channels at 30x (real
transformer activations have such channels; per-token scales don't help with them, a per-tensor
scale suffers more). A granularity that _scaled_mm rejects on this GPU is recorded as unsupported.

usage: python bench/fp8_micro.py --out results/fp8_micro.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from fp8_linear import quantize, scaled_mm  # noqa: E402


def time_ms(fn, iters=20):
    for _ in range(3):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[14040, 32760])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    shapes = [("qkvo", 1536, 1536), ("ffn_up", 1536, 8960), ("ffn_down", 8960, 1536)]

    def fp8_full(x, wq, ws):
        xq, xs = quantize(x, True)
        return scaled_mm(xq, xs, wq, ws)
    fp8_full_c = torch.compile(fp8_full)

    rows = []
    for m in args.tokens:
        for name, k, n in shapes:
            w = (torch.randn(n, k, device="cuda") * k ** -0.5).to(torch.bfloat16)
            flops = 2 * m * k * n
            for kind in ("gaussian", "outliers"):
                x = torch.randn(m, k, device="cuda")
                if kind == "outliers":
                    x[:, torch.randperm(k)[:8]] *= 30
                x = x.to(torch.bfloat16)
                ref = x.float() @ w.float().t()
                err = lambda y: ((y.float() - ref).norm() / ref.norm()).item()
                timed = kind == "gaussian"
                res = {}

                res["bf16"] = (err(F.linear(x, w)), time_ms(lambda: F.linear(x, w)) if timed else None)
                for label, rowwise, fast in [("fp8_tensorwise", False, False), ("fp8_rowwise", True, False),
                                             ("fp8_rowwise_fastacc", True, True)]:
                    try:
                        xq, xs = quantize(x, rowwise)
                        wq, ws = quantize(w, rowwise)
                        ws = ws.reshape(1, -1).contiguous() if rowwise else ws
                        y = scaled_mm(xq, xs, wq, ws, fast)
                        res[label] = (err(y), time_ms(lambda: scaled_mm(xq, xs, wq, ws, fast)) if timed else None)
                    except RuntimeError as ex:
                        res[label] = ("unsupported: " + str(ex).splitlines()[0][:160], None)
                if isinstance(res["fp8_rowwise"][0], float):
                    wq, ws = quantize(w, True)
                    ws = ws.reshape(1, -1).contiguous()
                    res["fp8_rowwise_with_quant"] = (err(fp8_full_c(x, wq, ws)),
                                                     time_ms(lambda: fp8_full_c(x, wq, ws)) if timed else None)
                bf16_ms = res["bf16"][1]
                for label, (e, ms) in res.items():
                    r = dict(tokens=m, shape=name, k=k, n=n, inputs=kind, backend=label, rel_l2_err=e, ms=ms,
                             tflops=flops / ms / 1e9 if ms else None,
                             speedup_vs_bf16=bf16_ms / ms if ms and bf16_ms else None)
                    rows.append(r)
                    es = f"{e:.2e}" if isinstance(e, float) else e
                    print(f"M={m:6d} {name:8s} {kind:8s} {label:22s} err {es}"
                          + (f" | {ms:7.3f} ms {r['tflops']:6.1f} TFLOPS x{r['speedup_vs_bf16']:.2f}" if ms else ""))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
