# Rounding noise floor: predictions before the run (2026-11-04)

Written before running `colab/03_noise_floor.ipynb`.

## Why

Compiled outputs differ from eager by a mean PSNR of 25.4–29.5 dB at 33 frames and 21.6 dB at 81 frames, with LPIPS 0.022–0.101 (`results/L4_compile.md`). Eager is bit-for-bit deterministic, so all of that difference comes from fused math rounding differently, amplified over 50 steps.

From one compiled sample alone, there is no way to tell:
- **(a)** any change of this size in rounding would move the trajectory this far, so compile is as good as eager; or
- **(b)** compile changes the numerics more than ordinary rounding does.

The same question will come up for every later optimization (8-bit attention, FP8, caching). This experiment measures the yardsticks once.

## Runs

Eager only, 50 steps, CFG 5.0, 480×832. Only one thing changes per run, against R.

| Run | Attention in the transformer | Seed | Frames × prompts | Role |
|---|---|---|---|---|
| R33 `flash_33f` | default (flash), same code path as the baseline | 0 | 33 × 4 | reference; determinism check |
| E33 `efficient_33f` | SDPA memory-efficient (cutlass), forced only inside the transformer | 0 | 33 × 4 | rounding control |
| S33 `seed1_33f` | default (flash) | 1 | 33 × 4 | sample-to-sample scale |
| R81 `flash_81f` | default (flash) | 0 | 81 × 1 (fox) | reference |
| E81 `efficient_81f` | memory-efficient | 0 | 81 × 1 (fox) | rounding control |

The two attention kernels compute exact softmax attention and accumulate in fp32. They tile and order the sums differently, so the bf16 outputs differ in the last bit on some elements.

Compile also differs from eager at bf16-ulp level. It drops intermediate bf16 roundings inside fused chains: the q/k RMSNorm, RoPE, gated residuals and GELU. So I expect the two kinds of perturbation to be of similar size.

Away from saturation, the final divergence scales with the size of the perturbation. A perturbation twice as large costs about 6 dB of PSNR, so ±4 dB means "within about 1.6× in size".

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| N1 | R33 reproduces the 2026-11-03 and 2026-11-04 eager frames bit for bit (`results/eager_33f/frames_sha256.json`). R81 reproduces the 2026-11-03 81-frame fox (`results/base_81f/frames_sha256.json`). | any hash differs |
| N2 | E33 vs R33: each video's mean PSNR is within ±4 dB of the compile-vs-eager PSNR for the same video. The mean over the four videos is within ±2.5 dB of the compile mean (27.8 dB). | any video outside ±4 dB, or the mean outside ±2.5 dB |
| N3 | The prompt ranking matches compile: night market has the lowest E33-vs-R33 PSNR, and fox or leaf the highest. How much a trajectory amplifies a small push depends on the prompt, not on where the push comes from. | another prompt is lowest, or coast is highest |
| N4 | E81 vs R81: mean PSNR 17.6–25.6 dB (compile: 21.6 dB), lower than every 33-frame video | outside the range, or not below all four 33-frame values |
| N5 | S33 vs R33 (different seed): mean PSNR 9–16 dB and LPIPS ≥ 0.30 on every video. This is the "different sample" scale, so 25–30 dB means "same sample, nudged". | any video above 16 dB or below 0.30 LPIPS |
| N6 | Memory-efficient attention is slower than flash: the attention kernel takes 1.3–2.0× as long at 33 f (673 ms → 875–1,350 ms per forward), and the forward is 10–33% slower. Mean SM clock within ±3% of R33, since both kernels are compute-bound. | outside any of these ranges |
| N7 | CLIP score (ViT-L/14, ×100, mean over every 4th frame): \|E33 − R33\| < \|S33 − R33\| on at least 3 of the 4 prompts | fewer than 3 |

## What each outcome means

- **N2 holds:** compile is within the rounding noise floor. Pixel distance to eager cannot judge its quality, and the 81-frame tail defect is most likely sample variance. From then on, an optimization counts as "same sample" if its PSNR against the reference is no more than 4 dB below the floor for that video. Below that, it changes the sample and has to be judged on CLIP score across seeds, not on pixel distance.
- **E33 lands more than 4 dB closer than compile on most videos:** compile perturbs the numerics more than attention reordering does. The next check is compiling with `torch._inductor.config.emulate_precision_casts = True`, which keeps eager's intermediate bf16 roundings.
- **E33 lands much farther than compile:** attention rounding is the more sensitive part. That would be a warning for 8-bit attention.
