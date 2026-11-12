# Rounding noise floor on Colab L4 (2026-11-04)

Setup:
- All runs are eager, in one Colab session. Same software as the earlier runs: torch 2.11.0+cu130, diffusers 0.42.1, transformers 5.22.1.
- Each run changes one thing against the flash reference (seed 0): either the SDPA backend inside the transformer, or the seed.

Raw data:
- `flash_33f/`, `efficient_33f/`, `seed1_33f/`, `flash_81f/`, `efficient_81f/`.
- `*/compare.json` holds the comparisons against the flash reference of the same length.
- `*/frames_sha256.json` holds frame hashes.

Predictions: `../predictions_noisefloor_2026-11-04.md`.

## Measurements

How far each change moves the output. Every row is compared against eager flash with seed 0, at the same length.

| Change | 33 f mean PSNR, per video (fox / coast / leaf / market) | 33 f LPIPS | 81 f fox PSNR | 81 f LPIPS |
|---|---|---|---|---|
| None (rerun, new session) | bit-identical, 4 of 4 | 0 | bit-identical | 0 |
| Memory-efficient attention | 36.4 / 36.0 / 36.2 / 36.2 (mean **36.2**) | 0.006–0.011 | **38.9** | 0.004 |
| `torch.compile` (from `L4_compile.md`) | 29.5 / 27.2 / 29.2 / 25.4 (mean **27.8**) | 0.022–0.080 | **21.6** | 0.101 |
| Seed 1 instead of 0 | 13.0 / 9.6 / 13.0 / 11.3 (mean **11.7**) | 0.52–0.77 | | |

Mean absolute pixel difference at 33 f: 1.7–2.0 for memory-efficient attention, 4.6–5.7 for compile, 38–63 for the seed change. At 81 f: 0.74 for memory-efficient attention and 8.7 for compile.

Speed, memory-efficient attention against flash in the same session:

| | 33 f flash | 33 f memory-efficient | 81 f flash | 81 f memory-efficient |
|---|---|---|---|---|
| Mean forward | 2,015 ms | 2,503 ms (+24.2%) | 6,455 ms | 8,994 ms (+39.3%) |
| Wall time per video | 212.7 s | 261.6 s | 673.0 s | 926.8 s |
| Attention kernel (one forward) | 665 ms | 1,197 ms (1.80×) | 3,327 ms* | 5,898 ms (1.77×) |
| GEMM kernels | 551 ms | 512 ms (−7.1%) | 1,350 ms* | 1,261 ms (−6.6%) |
| Norm + elementwise | 784 ms | 785 ms | 1,844 ms* | 1,844 ms |
| Mean SM clock | 1,194 MHz | 1,277 MHz (+7.0%) | 1,211 MHz | 1,290 MHz (+6.5%) |

\* The 81 f flash run in this session had no kernel profile. These kernel times come from `eager_81f` (2026-11-04, same code path, bit-identical output to the baseline).

## Predictions

| # | Prediction | Result | Verdict |
|---|---|---|---|
| N1 | Flash reruns reproduce the earlier eager frames bit for bit | 4 of 4 at 33 f, 1 of 1 at 81 f, now across three sessions | ✓ |
| N2 | Memory-efficient vs flash within ±4 dB of compile vs eager per video; mean within 27.8 ± 2.5 dB | 6.9–10.9 dB above compile per video; mean 36.2 dB (+8.4) | ✗ |
| N3 | Night market lowest and fox or leaf highest, as with compile | fox highest (36.4) but coast lowest (36.0); spread across prompts only 0.4 dB, against 4.1 dB for compile | ✗ |
| N4 | 81 f memory-efficient 17.6–25.6 dB and below every 33 f video | 38.9 dB, above every 33 f video | ✗ |
| N5 | Different seed: PSNR 9–16 dB and LPIPS ≥ 0.30 on every video | 9.6–13.0 dB, LPIPS 0.52–0.77 | ✓ |
| N6 | Attention 1.3–2.0× slower, forward 10–33% slower, clock within ±3% | 1.80×, +24.2%, but clock +7.0% | ✗ (clock) |
| N7 | CLIP: \|efficient − flash\| < \|seed 1 − flash\| on ≥ 3 of 4 prompts | not measured: `clip_score.py` assumed `get_image_features` returns a tensor, and in transformers 5.x it returns an output object. Fixed, and mp4 input added. | — |

2 of 6 measured predictions hold.

## What the results show

**1. Compile's divergence is not rounding noise.**
- Swapping one exact attention kernel for another moves the output to 36 dB at 33 f and 39 dB at 81 f, with LPIPS ≤ 0.011. In `figures/noise_floor_market_last_frame.png`:
  - top row: flash, memory-efficient, compile, seed 1;
  - bottom row: absolute difference from flash, ×4.
  - The memory-efficient difference is fine speckle on edges and texture. The maps are made from the mp4s, which are only 32.6–34.9 dB from the exact frames, so part of this speckle is codec noise. The PSNR and LPIPS values come from `.npy` and are not affected.
  - The compile difference shows moved lanterns, a street light that is not in eager, and people in different places.
  - The seed change gives a different street.
- At 81 f the gap is largest. Flash and memory-efficient attention give the same fox in the same pose. The compiled fox has turned to face the other way and has the detached extra tail (`figures/noise_floor_81f_fox_last_frame.png`: flash, memory-efficient, compile).
- So the compiled 81-frame defect cannot be put down to "any rounding change would do this". Compile changes the numerics in a way that ordinary reordering does not.

**2. What was wrong in my reasoning.**
- I assumed both perturbations sit at bf16-ulp level and that how far the output moves depends mostly on the trajectory. The data contradicts both.
- The attention control barely depends on the prompt (0.4 dB spread). It also gets *closer* at 81 f, not farther, so longer videos do not amplify a random rounding change more.
- Compile's distance depends strongly on the prompt (4.1 dB spread) and grows sharply with length.
- That pattern fits a systematic change better than a random one. Rounding errors that vary in sign tend to cancel over 30 layers and 50 steps. A consistent bias adds up.
- The most likely source is Inductor dropping eager's intermediate bf16 roundings inside fused chains: by default `torch._inductor.config.emulate_precision_casts` is off. Each such change is small but always in the same direction for a given input.
- This is not yet tested. Removing those roundings could also make compile *more* accurate than eager. Distance to eager does not say which of the two is closer to the full-precision result.

**3. The noise floor for later experiments.**

| Distance from eager flash, same seed | Mean PSNR | LPIPS | Reading |
|---|---|---|---|
| Rounding floor (exact kernel swap) | 36–39 dB | ≤ 0.011 | same sample |
| Compile | 21.6–29.5 dB | 0.02–0.10 | same scene, changed details and motion |
| Different seed | 9.6–13.0 dB | 0.52–0.77 | different sample |

- From here on, a change counts as "same sample" if it stays within about 4 dB of the floor: ≥ 32 dB at 33 f, ≥ 35 dB at 81 f.
- Compile does not meet that bar. Its quality has to be judged on its own: either against a full-precision reference, or on prompt-level metrics across seeds. Distance to eager does not settle it.

**4. The power cap, from the other side.**
- Memory-efficient attention is 1.80× slower than flash; normalized to clock it is about 1.93× slower. It is a less dense kernel, so it draws less power per unit time and the clock rises 7%.
- The GEMMs then run 7% faster. The elementwise kernels do not change at all (784 vs 785 ms), as expected for memory-bound work.
- This mirrors the compile run, where removing memory-bound time lowered the clock and slowed the GEMMs.
- On this power-capped L4, the clock follows the power density of the kernel mix, and the GEMMs follow the clock.
- N6 assumed both attention kernels draw the same power, which is why it missed.
- Flash is the right default; memory-efficient attention costs 24–39% per forward.

## Next

- **Find what compile changes.** Two runs would settle it:
  - Compile with `emulate_precision_casts=True`. If this lands near 36 dB of eager, the dropped roundings explain the divergence.
  - A full-precision transformer (fp32 weights and activations, TF32 off, memory-efficient attention since flash has no fp32 kernel) on two or more prompts. Then measure whether eager bf16 or compile bf16 is closer to it. This reference would also serve every later optimization.
- **Re-run CLIP** (N7) from the committed mp4s, together with the next notebook.
- **8-bit attention** can use the floor above directly: ≥ 32 dB at 33 f means the same sample; anything lower needs the full-precision reference or CLIP across seeds.
