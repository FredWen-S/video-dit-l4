#!/usr/bin/env python3
"""Reference-free quality metrics per video, from the exact frames (frames_<key>.npy).

These don't need a reference video, so they can compare runs that follow different sampling
trajectories (fewer steps, step caching), where PSNR to a 50-step reference only says "different".

Per video:
  clip_text    CLIPScore against the prompt (100 * cos, mean over frames), as in clip_score.py
  aesthetic    LAION aesthetic predictor v2 (sac+logos+ava1-l14-linearMSE) on the same CLIP ViT-L/14
               image embeddings, L2-normalized; mean over frames. Roughly a 1-10 scale.
  clip_consist mean cosine similarity of CLIP embeddings of consecutive scored frames (temporal
               consistency; 1 = identical frames)
  sharpness    variance of the Laplacian of luma, mean over all frames. Depends strongly on content
               and framing; read it as paired differences over many seeds only
  motion       mean absolute difference between consecutive frames (0-255). Descriptive, not a
               quality score: it checks whether a change makes videos more static

Every --every-th frame is scored by CLIP (default 2: 17 of 33 frames). The CLIP processor
resizes the short side to 224 and center-crops, so the left and right edges of a 832x480 frame
are not seen by the CLIP-based metrics.

Writes quality.json into each run directory.

usage: python bench/quality_metrics.py results/q_steps50 results/q_steps20 ...
"""
import argparse
import hashlib
import json
import os
import urllib.request

import numpy as np
import torch

AESTHETIC_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
                 "sac%2Blogos%2Bava1-l14-linearMSE.pth")


def aesthetic_head(dev):
    path = os.path.expanduser("~/.cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(AESTHETIC_URL, path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    # Same layout as the original MLP (Linear/Dropout pairs, no activations), so the state dict loads as is
    nn = torch.nn
    head = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
                         nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))
    sd = torch.load(path, map_location="cpu")
    head.load_state_dict({k.removeprefix("layers."): v for k, v in sd.items()})
    return head.to(dev).eval(), sha


def luma(frames):
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def sharpness(y):
    lap = 4 * y[:, 1:-1, 1:-1] - y[:, :-2, 1:-1] - y[:, 2:, 1:-1] - y[:, 1:-1, :-2] - y[:, 1:-1, 2:]
    return float(lap.var(axis=(1, 2)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--every", type=int, default=2)
    args = ap.parse_args()

    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(args.model)
    head, head_sha = aesthetic_head(dev)

    for run in args.runs:
        with open(os.path.join(run, "report.json")) as f:
            report = json.load(f)
        rows = []
        for i, r in enumerate(report["runs"]):
            key = r.get("key", r.get("prompt_id", i))
            frames = np.load(os.path.join(run, f"frames_{key}.npy"))
            scored = frames[::args.every]
            inp = proc(text=[r["prompt"]], images=list(scored), return_tensors="pt", padding=True,
                       truncation=True).to(dev)
            with torch.no_grad():
                img = model.get_image_features(pixel_values=inp["pixel_values"])
                txt = model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
                img, txt = (x if torch.is_tensor(x) else x.pooler_output for x in (img, txt))
                img = (img / img.norm(dim=-1, keepdim=True)).float()
                txt = (txt / txt.norm(dim=-1, keepdim=True)).float()
                clip_text = (100 * (img @ txt.T).squeeze(1).clamp(min=0)).mean().item()
                aesthetic = head(img).squeeze(1).mean().item()
                consist = (img[1:] * img[:-1]).sum(-1).mean().item()
            y = luma(frames)
            rows.append(dict(key=str(key), prompt_id=r.get("prompt_id", i), seed=r.get("seed", report["config"]["seed"]),
                             prompt=r["prompt"], clip_text=clip_text, aesthetic=aesthetic, clip_consist=consist,
                             sharpness=sharpness(y), motion=float(np.abs(np.diff(y, axis=0)).mean()),
                             frames_scored=len(scored)))
            print(f"{os.path.basename(run.rstrip('/'))} {key}: clip {clip_text:.2f} aes {aesthetic:.3f} "
                  f"consist {consist:.4f} sharp {rows[-1]['sharpness']:.1f} motion {rows[-1]['motion']:.2f}")
        with open(os.path.join(run, "quality.json"), "w") as f:
            json.dump(dict(model=args.model, every=args.every, aesthetic_weights_sha256=head_sha, source="npy",
                           steps=report["config"]["steps"], cache_threshold=report["config"].get("cache_threshold"),
                           videos=rows), f, indent=1)


if __name__ == "__main__":
    main()
