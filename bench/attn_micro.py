#!/usr/bin/env python3
"""Attention kernel microbenchmark at Wan2.1-1.3B shapes: flash (bf16) vs SageAttention variants.

Shapes: B=1, H=12, D=128, self-attention over N tokens (14,040 = 33 frames, 32,760 = 81 frames
at 480x832). For each backend: time per call (CUDA events, after warmup), effective TFLOPS
(4*N*N*D*H per call) and the relative L2 error of the output against an fp32 reference
(memory-efficient SDPA in fp32). Two input kinds: plain Gaussian q/k/v, and the same with a
per-channel bias added to K (real K often has one; softmax ignores it, 8-bit quantization does not
unless K is smoothed).

Random inputs say nothing about end-to-end quality; this is a kernel-level speed and sanity check.

usage: python bench/attn_micro.py --out results/sage_micro.json
"""
import argparse
import json

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def backends():
    out = {"flash": None}
    try:
        import sageattention as sa
        out["sage"] = sa.sageattn                                 # sm_89: INT8 QK, FP8 PV
        out["sage_fp16pv"] = sa.sageattn_qk_int8_pv_fp16_cuda     # INT8 QK, fp16 PV (fp32 accumulation)
    except ImportError as e:
        print(f"sageattention not importable: {e}")
    return out


def run(name, fn, q, k, v):
    """q, k, v in NHD layout (B, N, H, D), bf16. Returns the output in NHD."""
    if name == "flash":
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                  v.transpose(1, 2)).transpose(1, 2)
    return fn(q, k, v, tensor_layout="NHD", is_causal=False)


def reference(q, k, v):
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(*(t.float().transpose(1, 2) for t in (q, k, v))).transpose(1, 2)


def time_ms(fn, iters):
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
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--k-bias", type=float, default=4.0, help="std of the per-channel K bias")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    bk = backends()
    rows = []
    for n in args.tokens:
        q, k, v = (torch.randn(1, n, args.heads, args.dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        flops = 4 * n * n * args.dim * args.heads
        for kind in ("gaussian", "k_bias"):
            kk = k + args.k_bias * torch.randn(1, 1, args.heads, args.dim, device="cuda") if kind == "k_bias" else k
            kk = kk.to(torch.bfloat16)
            ref = reference(q, kk, v)
            for name, fn in bk.items():
                out = run(name, fn, q, kk, v).float()
                err = ((out - ref).norm() / ref.norm()).item()
                ms = time_ms(lambda: run(name, fn, q, kk, v), args.iters) if kind == "gaussian" else None
                r = dict(tokens=n, inputs=kind, backend=name, rel_l2_err=err, ms=ms,
                         tflops=flops / ms / 1e9 if ms else None)
                rows.append(r)
                print(f"N={n:6d} {kind:8s} {name:12s} rel err {err:.2e}"
                      + (f" | {ms:7.2f} ms {r['tflops']:6.1f} TFLOPS" if ms else ""))
            del ref
        flash_ms = next(r["ms"] for r in rows if r["tokens"] == n and r["backend"] == "flash" and r["ms"])
        for r in rows:
            if r["tokens"] == n and r["ms"]:
                r["speedup_vs_flash"] = flash_ms / r["ms"]
        print("  speedup vs flash: " + ", ".join(
            f"{r['backend']} {r['speedup_vs_flash']:.2f}x" for r in rows if r["tokens"] == n and r["ms"]))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
