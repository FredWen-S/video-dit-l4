#!/usr/bin/env python3
"""Hashes each frames_<i>.npy in a run directory, so determinism can be checked without the .npy.

Writes frames_sha256.json into the directory. With --check REF.json, also compares against the
hashes of an earlier run (only videos present in both) and exits non-zero on a mismatch.

usage: python bench/hash_frames.py results/flash_33f --check results/eager_33f/frames_sha256.json
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--check", help="frames_sha256.json of a reference run")
    args = ap.parse_args()

    hashes = {}
    for path in sorted(glob.glob(os.path.join(args.run, "frames_*.npy"))):
        a = np.ascontiguousarray(np.load(path))
        hashes[os.path.basename(path)] = dict(sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                                              shape=list(a.shape), dtype=str(a.dtype))
    if not hashes:
        raise SystemExit(f"no frames_*.npy in {args.run}")
    with open(os.path.join(args.run, "frames_sha256.json"), "w") as f:
        json.dump(hashes, f, indent=1)
    for k, v in hashes.items():
        print(f"{k}: {v['sha256'][:16]}")

    if args.check:
        with open(args.check) as f:
            ref = json.load(f)
        common = sorted(set(ref) & set(hashes))
        same = [k for k in common if ref[k]["sha256"] == hashes[k]["sha256"]]
        print(f"vs {args.check}: {len(same)} of {len(common)} identical"
              + (f", differ: {sorted(set(common) - set(same))}" if len(same) < len(common) else ""))
        if not common or len(same) < len(common):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
