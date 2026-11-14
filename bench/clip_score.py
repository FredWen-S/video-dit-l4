#!/usr/bin/env python3
"""CLIP score of each saved video against its prompt.

For each run directory, reads the prompts from report.json and the frames, embeds every
--every-th frame and the prompt with CLIP, and reports 100 * max(cos, 0) averaged over those
frames (the usual CLIPScore). Writes clip.json into each directory and prints a prompt-by-run table.

All runs are scored from the same source. The saved mp4s are only 32.6-34.9 dB from the exact
frames, with a mean color shift of up to 5 levels, and that moved CLIP by up to 1.5 points
(2026-11-11). So mixing .npy runs with mp4 runs compares the codec, not the runs.
--source auto uses .npy when every video has it, otherwise mp4 for all.

usage: python bench/clip_score.py results/flash_33f results/efficient_33f results/seed1_33f
"""
import argparse
import json
import os

import numpy as np
import torch


def read_mp4(path):
    """Decoded frames of a saved video, (F, H, W, 3) uint8."""
    import imageio
    with imageio.get_reader(path, "ffmpeg") as r:
        return np.stack([f for f in r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--every", type=int, default=4, help="score every n-th frame")
    ap.add_argument("--source", choices=["auto", "npy", "mp4"], default="auto")
    args = ap.parse_args()

    jobs = []
    for run in args.runs:
        with open(os.path.join(run, "report.json")) as f:
            jobs.append((run, [(r.get("key", r.get("prompt_id", i)), r["prompt"])
                               for i, r in enumerate(json.load(f)["runs"])]))  # key: multi-seed runs
    source = args.source
    if source == "auto":
        has_npy = all(os.path.exists(os.path.join(run, f"frames_{i}.npy")) for run, ps in jobs for i, _ in ps)
        source = "npy" if has_npy else "mp4"
    print(f"scoring from {source} for all runs")

    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(args.model)

    table = {}
    for run, prompts in jobs:
        rows = []
        for i, prompt in prompts:
            path = os.path.join(run, f"frames_{i}.npy" if source == "npy" else f"video_{i}.mp4")
            if not os.path.exists(path):
                raise SystemExit(f"missing {path} (--source {source})")
            frames = np.load(path) if source == "npy" else read_mp4(path)
            frames = frames[::args.every]
            inp = proc(text=[prompt], images=list(frames), return_tensors="pt", padding=True,
                       truncation=True).to(dev)
            with torch.no_grad():
                img = model.get_image_features(pixel_values=inp["pixel_values"])
                txt = model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
            # transformers 5.x returns BaseModelOutputWithPooling with the projection in pooler_output
            img, txt = (x if torch.is_tensor(x) else x.pooler_output for x in (img, txt))
            img = img / img.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            per_frame = (100 * (img @ txt.T).squeeze(1).clamp(min=0)).cpu().numpy()
            rows.append(dict(video=f"prompt_{i}", source=os.path.basename(path), prompt=prompt, clip=float(per_frame.mean()),
                             clip_min=float(per_frame.min()), frames_scored=len(per_frame)))
        with open(os.path.join(run, "clip.json"), "w") as f:
            json.dump(dict(model=args.model, every=args.every, source=source, videos=rows), f, indent=1)
        table[run] = {r["video"]: r["clip"] for r in rows}

    videos = sorted({v for t in table.values() for v in t})
    names = [os.path.basename(r.rstrip("/\\")) for r in args.runs]
    print("video".ljust(14) + "".join(n.rjust(16) for n in names))
    for v in videos:
        print(v.ljust(14) + "".join((f"{table[r][v]:.2f}" if v in table[r] else "-").rjust(16) for r in args.runs))


if __name__ == "__main__":
    main()
