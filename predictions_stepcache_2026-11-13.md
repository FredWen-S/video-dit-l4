# Step caching (first-block cache) on top of compile + Sage fp16-PV: predictions before the run (2026-11-13)

Written before running `colab/06_step_cache.ipynb`.

## The change

A first-block cache (`bench/step_cache.py`, `--cache-threshold`) skips whole forwards. Every forward runs the patch embedding and block 0. If block 0's residual has changed by less than the threshold since the last fully computed forward of the same CFG branch (relative mean absolute difference), blocks 1–29 are skipped. In that case the block-0 output is reused, plus the residual that blocks 1–29 produced last time they ran.

- **Rule:** the same as diffusers' `FirstBlockCache`.
- **Implementation:** an eager wrapper around the compiled blocks. diffusers' hooks would break fullgraph compile.
- **Thresholds swept:** 0.05 (diffusers' default), 0.10 and 0.20, at 33 frames with 4 prompts. 0.10 is also run at 81 frames on the fox.

**Base.** Compile plus SageAttention with INT8 QK and **fp16 PV** (`_sage_qk_int8_pv_fp16_cuda`). Its distance to fp32 matches compile's (35.4 / 26.6 dB), so any quality loss measured here comes from caching. The FP8-PV variant is faster but already uses up most of the market scene's quality margin.

**Control: fewer steps at the same compute.** A cache that computes 60% of the forwards is compared with simply running 30 steps (60% of the forwards). One that computes 40% is compared with 20 steps. If caching can't beat fewer steps at equal compute, it isn't worth having.

Caveat: every pixel metric here uses a 50-step reference (the uncached run, or fp32 at 50 steps). This favors caching, which approximates that same 50-step trajectory, over fewer steps, which follows a different discretization of the same ODE. CLIP score, which has no reference trajectory, is reported alongside.

## Starting point

Measured in `results/L4_sage.md`, at 33 f with Sage fp16-PV:

| | |
|---|---|
| Mean forward | 1,365 ms |
| Transformer per video | 136.5 s (100 forwards) |
| VAE decode | 10.9 s |
| Wall per video | 147.8 s |
| Distance to fp32 | fox 35.4 dB, market 26.6 dB |

**Bar against fp32**, as before: no worse than bf16 eager minus 1 dB. That is fox ≥ 27.1 dB and market ≥ 22.8 dB.

**Cost model of a skipped forward.** A skipped forward still runs:
- the patch embedding, timestep and text conditioning, and RoPE;
- block 0, which is 1/30 of the block time (about 45 ms);
- the output norm and projection;
- three or four fp32 passes over a 14,040 × 1,536 tensor, plus one `.item()` sync.

That totals about 4% of a full forward, call it k = 0.03–0.08. Computed forwards also pay the fp32 residual bookkeeping, under 1%. If s is the fraction of forwards skipped, the transformer time per video becomes T₀ · ((1 − s) + k·s).

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| C1 | The uncached fp16-PV run reproduces `sage_fp16pv_33f` bit for bit. Same torch, diffusers and SageAttention commit (pinned to d1a57a5) | any hash differs |
| C2 | Fraction of the 100 forwards skipped, mean over the 4 videos: 0.05 → 10–35%; 0.10 → 30–55%; 0.20 → 45–70% | any outside |
| C3 | Transformer time per video matches T₀ · ((1 − s) + k·s) with k between 0.03 and 0.08, using each video's own s. Equivalently, it is within the range those two values of k give | any video outside |
| C4 | PSNR against the uncached run, every video: 0.05 ≥ 26 dB; 0.10 in 20–30 dB; 0.20 in 15–25 dB | any video outside |
| C5 | Against fp32: 0.05 and 0.10 meet the bar on both fox and market; 0.20 fails it on at least one | any of these the other way |
| C6 | At matched compute, caching beats fewer steps. Pair the threshold whose computed fraction is closest to 60% with 30 steps, and the one closest to 40% with 20 steps. In each pair, the cache's mean PSNR to the uncached 50-step run is at least 2 dB higher. Pairs whose computed fractions differ by more than 10 points are not judged | either judged pair goes the other way or is closer than 2 dB |
| C7 | When skips happen. No threshold skips anything in steps 0–4, where the latent changes fastest. In the last third of the steps (34–49), the skip rate is at least that of the first third (0–16) | a skip in steps 0–4, or the first third skipping more |
| C8 | 81 f fox at 0.10. The skipped fraction is within 10 points of the 33 f mean at 0.10. PSNR to the uncached 81 f run is 0–5 dB below the 33 f mean at 0.10 | either outside |
| C9 | CLIP score (from npy, same source for every run), mean over the 4 prompts: every cached and fewer-step run is within ±0.5 of the uncached run | any outside |

## What each outcome means

- **C2, C4 and C5 hold, and C6 holds.** Caching at 0.10 is a real speedup of about 1.5×, with a known quality cost. It goes on top of the attention change. This would be the largest single speedup in the project.
- **C6 fails.** Running fewer steps is as good as caching, and simpler. The cache is dropped, and the step count becomes the knob.
- **C3 fails high, so k is larger than 0.08.** Skipped forwards cost more than one block. The likely culprits are the `.item()` sync or the fp32 residual passes. The fix is a cheaper decision step.
