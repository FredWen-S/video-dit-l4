#!/usr/bin/env python3
"""CLIP score of each saved video against its prompt.

For each run directory, reads the prompts from report.json and frames_<i>.npy, embeds every
--every-th frame and the prompt with CLIP, and reports 100 * max(cos, 0) averaged over those
frames (the usual CLIPScore). Writes clip.json into each directory and prints a prompt-by-run table.

usage: python bench/clip_score.py results/flash_33f results/efficient_33f results/seed1_33f
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--every", type=int, default=4, help="score every n-th frame")
    args = ap.parse_args()

    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(args.model)

    table = {}
    for run in args.runs:
        with open(os.path.join(run, "report.json")) as f:
            prompts = [r["prompt"] for r in json.load(f)["runs"]]
        rows = []
        for i, prompt in enumerate(prompts):
            path = os.path.join(run, f"frames_{i}.npy")
            if not os.path.exists(path):
                continue
            frames = np.load(path)[::args.every]
            inp = proc(text=[prompt], images=list(frames), return_tensors="pt", padding=True,
                       truncation=True).to(dev)
            with torch.no_grad():
                img = model.get_image_features(pixel_values=inp["pixel_values"])
                txt = model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
            img = img / img.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            per_frame = (100 * (img @ txt.T).squeeze(1).clamp(min=0)).cpu().numpy()
            rows.append(dict(video=os.path.basename(path), prompt=prompt, clip=float(per_frame.mean()),
                             clip_min=float(per_frame.min()), frames_scored=len(per_frame)))
        with open(os.path.join(run, "clip.json"), "w") as f:
            json.dump(dict(model=args.model, every=args.every, videos=rows), f, indent=1)
        table[run] = {r["video"]: r["clip"] for r in rows}

    videos = sorted({v for t in table.values() for v in t})
    names = [os.path.basename(r.rstrip("/\\")) for r in args.runs]
    print("video".ljust(14) + "".join(n.rjust(16) for n in names))
    for v in videos:
        print(v.ljust(14) + "".join((f"{table[r][v]:.2f}" if v in table[r] else "-").rjust(16) for r in args.runs))


if __name__ == "__main__":
    main()
