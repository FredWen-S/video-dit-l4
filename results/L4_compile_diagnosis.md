# What compile changes, on Colab L4 (2026-11-11)

Setup:
- 33 frames, 50 steps, CFG 5.0, seed 0, all runs in one Colab session.
- The Colab image had moved to **torch 2.14.1+cu130** (earlier runs: 2.11.0). diffusers 0.42.1 is unchanged.

Raw data:
- `flash_33f_s2/`: this session's eager run. Its frames and mp4s are byte-identical to `flash_33f/`.
- `compile_rerun_33f/`, `compile_emul_33f/`, `fp32_33f/`.
- `*/compare.json` compares against eager.
- `*/compare_vs_fp32.json` compares against the fp32 transformer.

Predictions: `../predictions_compile_diag_2026-11-04.md`.

## Measurements

Distance from eager bf16, by mean PSNR over frames, per video (fox / coast / leaf / market):

| Run | PSNR vs eager | LPIPS |
|---|---|---|
| Compile, default (rerun on torch 2.14) | 29.5 / 27.4 / 29.3 / 25.4 (mean 27.9) | 0.022–0.077 |
| Compile with `emulate_precision_casts=True` | 35.9 / 36.2 / 38.3 / 32.5 (mean **35.7**) | 0.006–0.019 |
| Memory-efficient attention, eager (2026-11-04, the rounding floor) | 36.4 / 36.0 / 36.2 / 36.2 (mean 36.2) | 0.006–0.011 |

The torch 2.14 rerun lands within 0.2 dB of the torch 2.11 compile run on every video.

Distance from the fp32 reference. The fp32 run uses the same bf16 weights, upcast to fp32, with TF32 off:

| Run | Fox PSNR | Fox LPIPS | Market PSNR | Market LPIPS |
|---|---|---|---|---|
| Eager bf16 | 28.1 | 0.029 | 23.8 | 0.110 |
| Compile, default | **35.6** | **0.007** | **26.2** | **0.077** |
| Compile with emulated casts | 27.8 | 0.030 | 23.9 | 0.107 |

Speed, same session:

| | Eager bf16 | Compile | Compile, emulated casts | fp32 eager |
|---|---|---|---|---|
| Mean forward | 2,020 ms | 1,474 ms (**1.37×**) | 1,517 ms (1.33×) | 8,250 ms (4.09× slower) |
| Wall time per video | 213.3 s | 158.8 s | 163.0 s | 851.6 s |
| Mean SM clock | 1,188 MHz | 1,123 MHz | 1,129 MHz | 1,224 MHz |
| Transformer peak memory | 4.11 GB | 3.82 GB | 3.82 GB | 7.62 GB |

Kernel time for one forward:
- **Compile:** flash 714 ms, GEMM 591 ms, everything else 155 ms.
- **Compile with emulated casts:** flash 710 ms, GEMM 587 ms, everything else 199 ms. The q/k RMSNorm + RoPE kernel splits into two (22 + 7 ms → 33 + 31 ms).
- **fp32:** attention 4,077 ms (cutlass fp32, 6.1× flash bf16), SGEMM 3,389 ms (6.2× bf16), everything else 806 ms (eager bf16: 784 ms). The VAE decode peaks at 19.7 GB with the fp32 transformer resident, close to the 22 GiB the card offers.

## Predictions

| # | Prediction | Result | Verdict |
|---|---|---|---|
| D1 | Eager reproduces the earlier frames bit for bit | 4 of 4, and across a torch upgrade 2.11 → 2.14 | ✓ |
| D2 | Emulated casts bring compile to ≥ 32 dB from eager on every video | 32.5–38.3 dB, mean 35.7 against the 36.2 dB floor | ✓ |
| D3 | Emulated casts cost 0–5% per forward over default compile | +2.9% | ✓ |
| D4 | Eager vs fp32: 22–34 dB on both videos | 28.1 and 23.8 dB | ✓ |
| D5 | Compile is closer to fp32 than eager on both videos | 35.6 vs 28.1 (fox), 26.2 vs 23.8 (market) | ✓ |
| D6 | fp32 forward 3–6× slower than eager bf16 | 4.09× | ✓ |
| D7 | CLIP: kernel swap within the seed spread, compile within the seed spread | cannot be judged: see point 4 below | — |

6 of 6 measured predictions hold. That is the first experiment in this project where all of them did. D2's worst video (market, 32.5 dB) passed by only 0.5 dB.

## What the results show

**1. The whole compile divergence is cast elision.**
- In eager, diffusers' Wan block computes each op in fp32 and rounds the result to bf16 before the next op reads it.
- Inductor fuses those ops and keeps the intermediate in fp32. That drops a rounding at every fusion boundary, and the dropped roundings always push in the same direction for a given input.
- With `emulate_precision_casts=True`, Inductor keeps those roundings and compile lands at the rounding floor (35.7 against 36.2 dB).
- Nothing else that compile changes matters at this scale: reduction order in the norms, the GELU implementation, the bias handling.

**2. Compile is the more accurate path, not the less accurate one.**
- Against fp32, compile is 7.5 dB closer than eager on the fox (35.6 vs 28.1) and 2.4 dB closer on the night market.
- Emulating eager's casts moves compile back to eager's distance (27.8, 23.9).
- On the fox, compile is about as far from fp32 as the kernel swap is from eager.
- `figures/compile_diag_market_last_frame.png` shows the last frame of the night market: fp32, eager, compile, and compile with emulated casts. The street light that compile "added" in `L4_noise_floor.md` is in the fp32 frame too. Eager and emulated compile lack it.
- So the divergence that looked like a quality risk is compile tracking the full-precision model better than eager does.
- This is two videos at 33 frames. The 81-frame fox, where compile turned the fox around and added a tail, has no fp32 reference yet.

**3. Emulated casts have no use here.** They cost 2.9% per forward and give back the accuracy that compile gained. Default compile is both faster and closer to fp32.

**4. CLIP: the scores read from mp4 and from .npy are not comparable.**
- `clip_score.py` read this session's runs from `.npy` and the earlier runs from their mp4s.
- The saved mp4s (h264, yuv420p, about 1.4 Mbit/s) are only 32.6–34.9 dB from the exact frames, with a mean color shift of up to 5 levels in red.
- Frames that are nearly identical scored 0.0–1.5 points higher from mp4 than from `.npy`. That is as large as any difference between runs, so D7 compared the codec, not the runs.
- These scores are kept as `clip_mixed_sources.json` and are not used. `clip_score.py` now scores every run from one source.
- Among this session's `.npy` runs, the scores are:

| Prompt | Eager | Compile | Emulated | fp32 |
|---|---|---|---|---|
| Fox | 33.79 | 33.60 | 33.64 | 33.59 |
| Coast | 28.12 | 28.12 | 28.10 | |
| Leaf | 27.03 | 26.85 | 26.99 | |
| Market | 28.91 | 29.96 | 28.68 | 29.65 |

- Every same-seed variant is within 1.1 points of eager. On the market, compile and fp32 both score about 1 point above eager and emulated compile. That is the same grouping as the pixel distances, but it is a single sample.
- With one seed per prompt, CLIP cannot separate anything here.
- The codec loss also matters for `L4_noise_floor.md`. Its difference maps were made from mp4s, so part of the speckle in the memory-efficient column is codec noise. Its PSNR and LPIPS numbers came from `.npy` and are unaffected.

## What this changes

- **The quality reference is now fp32, not eager.** For the fox and the night market at 33 frames, an optimization is "no worse than the bf16 baseline" if its PSNR to fp32 is at least eager's (28.1 / 23.8 dB) minus 1 dB. Compile sets a higher bar: 35.6 / 26.2 dB.
- **Distance to eager is still a cheap first check.** At ≥ 32 dB the change is within rounding noise. Below that, compare against fp32 before calling the change a quality loss.
- **Pixel metrics must come from `.npy`.** The mp4s are for viewing.
- **Compile stays on as the new baseline:** 1.37× on torch 2.14 and more accurate.

## Next

- **8-bit attention** (SageAttention2 has an sm_89 path), on top of compile. Judge it against the fp32 reference (fox and market) and against compile's distance to eager.
- **Optional:** an 81-frame fp32 fox (about 35 min), to check whether the turned-around fox with the extra tail is also what fp32 produces.
