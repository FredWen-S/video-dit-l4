# Reference-free quality protocol: predictions before the run (2026-11-13)

Written before running `colab/07_quality.ipynb`.

## Why

`results/L4_step_cache.md` showed that fewer steps and step caching produce a *different* video, not a perturbed one. At 30 steps the output is already 16.6 dB from the 50-step output, which is near the distance of a different seed. PSNR to a 50-step reference flags that the sample changed but can't say whether it got worse. The fp32 bar stays for purely numerical changes (compile, attention kernels, quantization). Trajectory changes need a test that compares *distributions of videos* without a reference video.

## The protocol

**Videos.** 4 prompts × 4 seeds (0–3) = 16 videos per configuration, at 33 frames on compile + Sage fp16-PV. The configurations:

| Run | Configuration | Role |
|---|---|---|
| `q_steps50` | 50 steps, no cache | base |
| `q_steps20` | 20 steps | candidate (2.24× in `L4_step_cache.md`) |
| `q_steps10` | 10 steps | probing where fewer steps breaks |
| `q_fbc010` | 50 steps, first-block cache 0.10 | candidate (2.39×) |
| `q_fbc020` | 50 steps, first-block cache 0.20 | **positive control**: visibly broken on seed 0 |

Cache 0.05 is left out. It keeps the 50-step sample, so the existing PSNR comparison already applies to it.

**Metrics** (`bench/quality_metrics.py`), from the exact frames (npy), with no reference video:
- **clip_text:** CLIPScore against the prompt (ViT-L/14), mean over 17 of the 33 frames.
- **aesthetic:** LAION aesthetic predictor v2 on the same CLIP embeddings.
- **clip_consist:** mean cosine similarity of the CLIP embeddings of consecutive scored frames, a measure of temporal consistency.
- **sharpness:** variance of the Laplacian of luma.
- **motion:** mean absolute difference between frames. It is descriptive and gets no verdict.

**Statistics** (`tools/quality_table.py`):
- Each run is paired with the base by prompt and seed. Same prompt, same initial noise.
- The scale is the base's **seed SD**: the spread between seeds within a prompt. Changing the seed is a change that says nothing about quality.
- Verdicts use a 95% bootstrap CI of the mean paired difference and a margin of **1.0 seed SD**:
  - **not worse:** CI lower end above −1 SD;
  - **worse:** CI upper end below −1 SD;
  - **unclear:** anything else.
- Why 1.0 SD rather than something tighter: with 16 pairs of samples that share nothing but the prompt and seed, the CI half-width is about 0.7 SD. A 0.5 SD margin could not be met even by a configuration with exactly the base's quality.

**Blind side-by-side** (`tools/blind_ab.py`). After the run, this is judged locally by a person in a browser:
- 48 pairs: base against `q_steps20`, `q_fbc010` and `q_fbc020`, each on all 16 videos;
- left/right order randomized, run names hidden;
- answers are "left better", "same" or "right better".

The score is the share of pairs where the base is preferred, with ties counted as half.

**Limits.** 33 frames only, 4 prompts, and one person judging. CLIP sees only the 224-pixel center crop of each frame.

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| Q1 | Seed 0 reproduces the earlier runs bit for bit (16 videos): `q_steps50`, `q_steps20`, `q_fbc010`, `q_fbc020` vs `pv16_33f`, `steps20_33f`, `fbc010_33f`, `fbc020_33f` | any hash differs |
| Q2 | **The metrics can see a known break.** `q_fbc020` is "worse" on aesthetic, and on at least one of clip_text or sharpness | either not "worse". Then the automatic metrics are blind to the failure that matters, and the blind test decides alone |
| Q3 | `q_steps20` is "not worse" on all four judged metrics | any metric "worse" or "unclear" |
| Q4 | `q_steps10` is "worse" on aesthetic or sharpness | neither |
| Q5 | `q_fbc010` is "not worse" on clip_text and clip_consist. Its mean sharpness difference to the base is more negative than `q_steps20`'s, because the cache reuses stale residuals | either fails |
| Q6 | Motion: `q_fbc020` is more than 1 seed SD below the base; `q_steps20` is within ±1 SD | either outside |
| Q7 | Blind test, base preferred: vs `q_steps20` 35–65%; vs `q_fbc010` 35–65%; vs `q_fbc020` ≥ 80% | any outside |
| Q8 | Speed matches `L4_step_cache.md`, which used seed 0. Wall time per video within 3% for 50, 20 and 10 steps. For the caches, the fraction of computed forwards is within ±8 points of seed 0's mean (34% at 0.10, 18% at 0.20) | any outside |
| Q9 | Scale of the seed SD: clip_text 0.8–2.0 points; aesthetic 0.08–0.25 | either outside |

## What each outcome means

- **Q2 and Q3 hold.** 20 steps is as good as 50 by every measure the project has. 20 steps then joins the baseline, for 2.2× on top of compile + Sage. Step caching isn't needed unless Q5 shows an advantage it didn't show before.
- **Q2 fails.** CLIP-based metrics and sharpness can't see the 0.20 collapse. Only the blind test counts, and adding a no-reference image-quality model (e.g. MUSIQ) becomes the next step.
- **Q3 fails on sharpness or aesthetic.** 50 → 20 steps has a real cost. The question becomes where between 20 and 50 the metrics recover; that would be a 30- and 40-step run next.
