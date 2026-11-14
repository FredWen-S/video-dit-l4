#!/usr/bin/env python3
"""Paired comparison of reference-free quality metrics against a base run (numpy only, runs locally).

Reads quality.json (bench/quality_metrics.py) from each run. Videos are paired by key
(prompt and seed), so a run is compared with the base on the same prompt and the same initial noise.

For each metric:
- seed SD: the spread between seeds of the base run, pooled within prompts. Changing the seed is
  the size of change that says nothing about quality, so it sets the scale.
- margin m = 1.0 x seed SD (fixed in predictions_quality_2026-11-13.md before the run). With 16 pairs of
  uncorrelated samples the CI half-width is about 0.7 SD, so a tighter margin could not be met even by a
  run with exactly the base's quality distribution.
- mean paired difference (run - base) with a 95% bootstrap CI over the pairs (10,000 resamples).
- verdict: "not worse" if the CI's lower end is above -m; "worse" if its upper end is below -m;
  otherwise "unclear". motion has no verdict (more or less motion is not better or worse).

usage: python tools/quality_table.py --base results/q_steps50 results/q_steps20 results/q_fbc010 ...
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

METRICS = ["clip_text", "aesthetic", "clip_consist", "sharpness", "motion"]
JUDGED = {"clip_text", "aesthetic", "clip_consist", "sharpness"}


def load(run):
    with open(os.path.join(run, "quality.json")) as f:
        return {v["key"]: v for v in json.load(f)["videos"]}


def seed_sd(base, metric):
    by_prompt = defaultdict(list)
    for v in base.values():
        by_prompt[v["prompt_id"]].append(v[metric])
    devs = [x - np.mean(xs) for xs in by_prompt.values() if len(xs) > 1 for x in xs]
    dof = sum(len(xs) - 1 for xs in by_prompt.values() if len(xs) > 1)
    return float(np.sqrt(np.sum(np.square(devs)) / dof)) if dof else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--margin", type=float, default=1.0, help="margin in units of the base's seed SD")
    ap.add_argument("--out", default=None, help="write the table as JSON")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    base = load(args.base)
    out = {"base": args.base, "n_base": len(base), "metrics": {}}
    print(f"base {args.base}: {len(base)} videos")
    for m in METRICS:
        sd = seed_sd(base, m)
        bm = float(np.mean([v[m] for v in base.values()]))
        out["metrics"][m] = dict(base_mean=bm, seed_sd=sd, margin=args.margin * sd, runs={})
        print(f"\n{m}: base mean {bm:.4g}, seed SD {sd:.4g}, margin {args.margin * sd:.4g}")
        for run in args.runs:
            q = load(run)
            keys = sorted(set(q) & set(base))
            d = np.array([q[k][m] - base[k][m] for k in keys])
            boots = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            mg = args.margin * sd
            verdict = ("-" if m not in JUDGED else
                       "not worse" if lo > -mg else "worse" if hi < -mg else "unclear")
            out["metrics"][m]["runs"][run] = dict(n=len(d), mean_diff=float(d.mean()), ci=[float(lo), float(hi)],
                                                  diff_in_sd=float(d.mean() / sd) if sd else None, verdict=verdict)
            print(f"  {os.path.basename(run.rstrip('/')):18s} n={len(d):2d} diff {d.mean():+.4g} "
                  f"[{lo:+.4g}, {hi:+.4g}] = {d.mean() / sd:+.2f} SD  {verdict}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
