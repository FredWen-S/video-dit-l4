#!/usr/bin/env python3
"""Compares the frames of a test run against a reference run (same prompts, same seed).

For each frames_<i>.npy present in both directories, reports whether they are bit-identical,
max and mean absolute pixel difference, per-frame PSNR (mean and worst frame) and, if the
`lpips` package is installed, mean per-frame LPIPS (AlexNet). Writes compare.json to --test.

usage: python bench/compare_frames.py --ref results/eager_33f --test results/compile_33f
"""
import argparse
import glob
import json
import os

import numpy as np


def psnr(a, b):
    """Per-frame PSNR in dB of uint8 arrays shaped (F, H, W, 3); inf where identical."""
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
    with np.errstate(divide="ignore"):
        return 10 * np.log10(255.0 ** 2 / mse)


def lpips_fn():
    try:
        import lpips
        import torch
    except ImportError:
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    def score(a, b, batch=8):
        out = []
        with torch.no_grad():
            for i in range(0, len(a), batch):
                x, y = (torch.from_numpy(np.ascontiguousarray(t[i:i + batch])).to(dev)
                        .permute(0, 3, 1, 2).float() / 127.5 - 1 for t in (a, b))
                out.append(net(x, y).flatten().cpu())
        return torch.cat(out).numpy()
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    score = None if args.no_lpips else lpips_fn()
    rows = []
    for ref_path in sorted(glob.glob(os.path.join(args.ref, "frames_*.npy"))):
        test_path = os.path.join(args.test, os.path.basename(ref_path))
        if not os.path.exists(test_path):
            continue
        a, b = np.load(ref_path), np.load(test_path)
        assert a.shape == b.shape, (ref_path, a.shape, b.shape)
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        p = psnr(a, b)
        r = dict(video=os.path.basename(ref_path), identical=bool(d.max() == 0), max_abs=int(d.max()),
                 mean_abs=float(d.mean()), psnr_mean=float(np.mean(np.minimum(p, 100))),
                 psnr_min=float(p.min()))
        if score is not None and not r["identical"]:
            r["lpips_mean"] = float(score(a, b).mean())
        rows.append(r)
        print(f"{r['video']}: identical={r['identical']} max|d|={r['max_abs']} mean|d|={r['mean_abs']:.3f} "
              f"PSNR mean {r['psnr_mean']:.1f} dB (min {r['psnr_min']:.1f})"
              + (f" LPIPS {r['lpips_mean']:.4f}" if "lpips_mean" in r else ""))
    if not rows:
        raise SystemExit("no frames_*.npy present in both directories")
    with open(os.path.join(args.test, "compare.json"), "w") as f:
        json.dump(dict(ref=args.ref, test=args.test, videos=rows), f, indent=1)


if __name__ == "__main__":
    main()
