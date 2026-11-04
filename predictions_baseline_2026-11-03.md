# Baseline predictions before the first run (2026-11-03)

Written before running `bench/profile_baseline.py` on any GPU. Setup: Wan2.1-T2V-1.3B (diffusers), 480×832, 50 UniPC steps, CFG 5.0 (two transformer forwards per step), bf16 transformer and text encoder, fp32 VAE, PyTorch SDPA, Colab L4.

## Cost model (`tools/cost_model.py`)

The VAE compresses 4× in time and 8× in space and the DiT patchifies 1×2×2, so the token count is `((F-1)/4+1) · 30 · 52` at 480×832.

| Frames | Tokens N | Linear TFLOP / forward | Attention TFLOP / forward | Elementwise GB / forward (unfused) | Model share: attn / linear / elementwise | Est. denoising time |
|---|---|---|---|---|---|---|
| 81 | 32,760 | 82 | 201 | 120 | 65% / 24% / 11% | 7.8 min |
| 33 | 14,040 | 35 | 38 | 51 | 44% / 38% / 17% | 2.1 min |
| 17 | 7,800 | 20 | 12 | 29 | 31% / 47% / 22% | 1.0 min |

Assumptions: 121 TFLOPS dense bf16 peak, 55% achieved on attention, 60% on GEMMs, 231 GB/s for elementwise kernels, 14 unfused read+write passes over an N×1536 activation per block plus one GELU pass over N×8960.

Two things follow:
- **Unlike LLM decode, this workload is compute-bound.** Each linear layer sees M = N ≥ 7,800 rows, so weight-only int8/FP8 cannot speed it up. It only saves about 1.3 GB of weights. FP8 helps only as W8A8 on FP8 tensor cores, and only on the linear share.
- **The bottleneck depends on clip length.** Attention grows with N², linear layers with N. At 81 frames attention dominates; at 17 frames it does not. Results measured at the cheap 33-frame iteration setting will not rank optimizations the same way as at 81 frames.

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| B1 | Wall time per video: 33 frames 1.6–3.0 min, 81 frames 6–11 min | either outside its range |
| B2 | Transformer forwards take ≥ 90% of wall time at both lengths; text encoding < 1%; VAE decode 2–8% at 81 frames | any outside its range |
| B3 | Kernel split of one forward at 81 frames: attention 55–75%, GEMM 18–32%, norm + other 6–15% | any outside its range |
| B4 | At 33 frames the attention share is ≥ 15 pp lower than at 81 frames, and GEMM is 30–45% | attention gap < 15 pp, or GEMM outside 30–45% |
| B5 | Mean forward time, 81 / 33 frames: 3.0–4.2× (model: 3.7×), clearly above the 2.33× token ratio | outside 3.0–4.2× |
| B6 | Resident weights 14–15.5 GB (umT5-XXL encoder ≈ 11.4 GB is most of it); peak allocated < 22 GB at 81 frames, so no offload is needed on 24 GB | outside 14–15.5 GB, or peak ≥ 22 GB / OOM |
| B7 | Mean SM clock during the run ≤ 1,800 MHz (below the 2,040 MHz boost) because the card hits its 72 W cap | mean > 1,800 MHz |

Notes for reading the results:
- If B3 fails because attention is much lower, check which SDPA backend ran (the top-kernel list shows `flash` vs `efficient_attention` vs a math fallback).
- If B7 fails the other way (clocks far below 1,800 MHz), all achieved-TFLOPS numbers in later steps must be read against the measured clock, not the datasheet peak.
