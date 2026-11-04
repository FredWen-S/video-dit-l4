#!/usr/bin/env python3
"""Analytic cost model of one Wan2.1-T2V-1.3B transformer forward, written before any GPU run.

Counts matmul FLOPs (linear layers vs attention) and estimates elementwise bytes, then turns
them into time with an assumed achieved throughput on an L4. The point is to know, before
profiling, which part an optimization can touch at all (Amdahl), not to predict exact times.

usage: python tools/cost_model.py [--frames 81 33] [--height 480] [--width 832] [--steps 50]
"""
import argparse

# Wan2.1-T2V-1.3B transformer (check against transformer/config.json of the diffusers repo)
DIM, FFN, LAYERS, TEXT_LEN = 1536, 8960, 30, 512
VAE_T, VAE_S = 4, 8       # VAE downsamples time by 4 and space by 8
PATCH_T, PATCH_S = 1, 2   # DiT patchifies latents 1 x 2 x 2

# L4 assumptions. Dense peaks from the datasheet (sparsity numbers halved). Achieved fractions
# are guesses: the L4 is a 72 W card and drops clocks under sustained tensor-core load.
BF16_PEAK, FP8_PEAK = 121e12, 242e12
ATTN_EFF, GEMM_EFF = 0.55, 0.60
BW = 231e9                # copy bandwidth measured on Colab L4 in lowbit-decode-gpu
# Full read+write passes over an N x DIM bf16 activation per block when nothing is fused:
# norm1, modulate, gate+residual (self-attn); norm3, residual (cross-attn); norm2, modulate,
# gate+residual (FFN); q/k RMSNorm and RoPE on q and k. Rough, on purpose.
PASSES_D = 14
PASSES_FFN = 1            # GELU over N x FFN


def tokens(frames, h, w):
    t = (frames - 1) // VAE_T + 1
    return t * (h // VAE_S // PATCH_S) * (w // VAE_S // PATCH_S), t


def forward_cost(n):
    lin = 2 * n * (4 * DIM * DIM          # self-attn q, k, v, o
                   + 2 * DIM * DIM        # cross-attn q, o
                   + 2 * DIM * FFN)       # FFN up, down
    lin += 2 * TEXT_LEN * 2 * DIM * DIM   # cross-attn k, v on text tokens
    attn = 4 * n * n * DIM + 4 * n * TEXT_LEN * DIM   # QK^T and PV, self + cross
    elem_bytes = 2 * 2 * n * (PASSES_D * DIM + PASSES_FFN * FFN)   # read + write, bf16
    return lin * LAYERS, attn * LAYERS, elem_bytes * LAYERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[81, 33])
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    fwd_per_video = 2 * args.steps        # CFG: Wan's pipeline runs cond and uncond separately

    for f in args.frames:
        n, t = tokens(f, args.height, args.width)
        lin, attn, eb = forward_cost(n)
        t_lin, t_attn = lin / (BF16_PEAK * GEMM_EFF), attn / (BF16_PEAK * ATTN_EFF)
        t_elem = eb / BW
        tot = t_lin + t_attn + t_elem
        print(f"\n{f} frames {args.height}x{args.width}: latent T={t}, N={n} tokens")
        print(f"  per forward: linear {lin/1e12:.1f} TFLOP, attention {attn/1e12:.1f} TFLOP, "
              f"elementwise {eb/1e9:.0f} GB (unfused)")
        print(f"  time share:  attention {t_attn/tot:.0%}, linear {t_lin/tot:.0%}, elementwise {t_elem/tot:.0%}")
        print(f"  est. forward {tot:.2f} s -> denoising {tot*fwd_per_video/60:.1f} min "
              f"({fwd_per_video} forwards)")
        # Amdahl ceilings for single changes, assuming the touched part gets exactly its ideal factor
        def speedup(**k):
            new = t_attn / k.get("attn", 1) + t_lin / k.get("lin", 1) + t_elem / k.get("elem", 1)
            return tot / new
        print("  ceilings: "
              f"W8A8 FP8 linears (2x) {speedup(lin=2):.2f}x | "
              f"8-bit attention (2x) {speedup(attn=2):.2f}x | "
              f"weight-only int8/FP8 ~1.00x | "
              f"remove all elementwise traffic {speedup(elem=1e9):.2f}x | "
              f"fuse 2 of {PASSES_D} passes {speedup(elem=1/(1-2*DIM/(PASSES_D*DIM+PASSES_FFN*FFN))):.3f}x")


if __name__ == "__main__":
    main()
