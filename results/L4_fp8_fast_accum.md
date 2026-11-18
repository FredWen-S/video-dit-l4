# FP8 rowwise with fast accumulation, fp32 references for all four videos, activation statistics: L4 (2026-11-18)

Predictions: `predictions_fp8fast_2026-11-18.md`. Notebook: `colab/09_fp8_fast_accum.ipynb`. Follows `L4_fp8_linears.md`.

**Result: 3 of 9 predictions held.**

**Speed.** FP8 rowwise with fast accumulation is the first FP8 linear variant that pays off in speed:
- **1.17×** faster at 33 frames, **1.10×** at 81 frames;
- 1.1–1.3 GB less memory.

**Quality.** It misses the fp32 bar on all four videos, and on every one it is further from fp32 than bf16 eager is.

**What these runs teach about the measurement, which matters more than the speedup:**

- **Fast accumulation vs plain rowwise.** Videos 0 and 3 got worse, by 6.9 dB and 1.3 dB. I can't separate that from noise.
  - At the GEMM level, fast and plain accumulation have the same error to three digits.
  - Against the base, the per-video distances of the two variants swap by up to 7 dB in both directions: the fox gets worse, the leaf better.
  - At FP8's size of perturbation, a single video's PSNR to a reference is a noisy draw. It cannot rank FP8 variants.
- **Plain rowwise's pass last time.** Last time, plain rowwise met the bar on the fox by 0.13 dB. That now looks like a favorable draw, not evidence that it is more accurate.
- **The base on the coast.** The quality-clean base itself, compile + Sage fp16-PV, misses the per-video bar on the coast: 25.7 dB against a bar of 27.7. It is closer to fp32 than eager on the other three videos and on average.
- **Tensorwise's 6 dB loss.** The activation statistics show tensorwise is only 0.2–9% worse than rowwise per GEMM. So last time's 6 dB gap on the fox is most likely the same noise, not a defect of per-tensor scales.

## Setup

- Same environment, again logged:
  - torch 2.14.1+cu130, diffusers 0.42.1, SageAttention 2.2.0, Triton 3.8.0;
  - driver 580.82.07, nvcc 13.0.
- **Base:** per-block fullgraph compile + Sage INT8-QK / fp16-PV, 50 steps, CFG 5.0, seed 0.
- **FP8 run:** `--fp8-linears rowwise --fp8-fast-accum`, 300 linears replaced.
- **fp32 reference:** all four prompts.
- **Eager bf16:** coast and leaf, which set the bars for those two videos.
- **Activation statistics:** eager bf16 + Sage fp16-PV, fox and market. Sampled at steps 0, 20 and 49, both CFG branches, for all 300 linears.

## Verdicts

| # | Prediction | Measured | |
|---|---|---|---|
| G1 | 33 f base = `f_base_33f` = `pv16_33f`; eager = `flash_33f`; fp32 = `fp32_33f` | 4 of 4, 4 of 4, 2 of 2, 2 of 2 identical | ✓ |
| G2 | 81 f base = `f_base_81f` (same flags), ≠ `pv16_81f` | the reverse: **= `pv16_81f`**, ≠ `f_base_81f` | ✗ |
| G3 | 33 f: 1.06–1.15× | **1.17×** (1,352 → 1,158 ms) | ✗ |
| G4 | 33 f: GEMM 320–420 ms; norm + other +60–120 ms; clock ±5% | GEMM 598 → 346 ms; +96 ms; clock **+7.2%** | ✗ |
| G5 | 81 f: 1.04–1.12× | 1.10× (4,444 → 4,025 ms) | ✓ |
| G6 | vs fp32 within 0.5 dB of plain rowwise: fox 26.7–27.7, market 23.0–24.0 | fox **20.33**, market **22.16** | ✗ |
| G7 | eager vs fp32: coast and leaf 24–30 dB | 28.65 and 27.34 | ✓ |
| G8 | meets the bar on 0 and 3, misses on 1 and 2 by ≥ 2 dB | misses on **all four**; on the leaf by only 0.44 dB | ✗ |
| G9 | `tr` ≈ `tt` and `rt` ≈ `rr` (within 20%); the ten worst layers by `tt / rr` have token spread > 30 | the first part holds, but the whole gap is 1–2%; the worst layers have token spread **1.0–1.3** | ✗ |

**On G3 and G4.** The GEMM landed where the microbenchmark put it (346 ms). The extra speed came from the clock:
- The SM clock rose 7%, from 1,111 to 1,191 MHz.
- The unchanged Sage attention kernel got 41 ms faster (543 → 502 ms).
- Without that, the speedup would be about 1.13×.

Tensorwise last time had similar GEMM speed and left the clock unchanged. So the two FP8 GEMM kernels (cuBLASLt tensorwise vs CUTLASS rowwise) apparently draw different power. I did not measure power per kernel.

**On G2.** With identical flags, the 81 f base now matches `pv16_81f` (2026-11-13) and not `f_base_81f` (earlier today). So 2 of 3 sessions agree, and the profiler is not the cause. There is rare run-to-run nondeterminism, at the size of rounding noise. It has now appeared once at 81 f and once at 33 f (`q_steps50`), in about six sessions. The cause is unknown. Cross-session comparisons need PSNR as well as hashes.

## Speed

| | Mean forward | vs base | SM clock | Peak memory | Wall per video |
|---|---|---|---|---|---|
| 33 f base | 1,352 ms | 1.00× | 1,111 MHz | 3.82 GB | 146.4 s |
| 33 f FP8 rowwise + fast acc. | 1,158 ms | **1.17×** | 1,191 MHz | 2.55 GB | 127.1 s |
| 81 f base | 4,444 ms | 1.00× | 1,176 MHz | 4.35 GB | 471.8 s |
| 81 f FP8 rowwise + fast acc. | 4,025 ms | **1.10×** | 1,220 MHz | 3.25 GB | 429.9 s |

**Kernel time per forward** (ms):

| | GEMM | Attention | Attention quantization | Norm | Other | Total |
|---|---|---|---|---|---|---|
| 33 f base | 598 | 543 | 22 | 80 | 103 | 1,345 |
| 33 f fast acc. | 346 | 502 | 21 | 118 | 161 | 1,148 |
| 81 f base | 1,357 | 2,618 | 57 | 203 | 250 | 4,484 |
| 81 f fast acc. | 759 | 2,544 | 56 | 293 | 378 | 4,029 |

**Where the time went:**
- FP8 GEMMs run at 1.73–1.79× bf16 in the model.
- The fused activation quantization costs 96 ms at 33 f and 218 ms at 81 f, as in the previous run.
- At 81 f, attention is 63% of the forward after this change. FP8 linears have little left to give there.

## Quality against fp32

Mean PSNR over frames, from npy, with LPIPS in parentheses. The bar is eager's distance minus 1 dB:
- fox and market: the fixed values set on 2026-11-11;
- coast and leaf: from this session's eager run.

| vs fp32 | Fox | Coast | Leaf | Market | Mean |
|---|---|---|---|---|---|
| Eager bf16 | 28.1 | 28.65 (0.049) | 27.34 (0.055) | 23.8 | 26.97 |
| **Bar** | 27.1 | 27.65 | 26.34 | 22.8 | |
| Base: compile + Sage fp16-PV | 35.39 (0.007) | **25.73** (0.074) | 32.57 (0.021) | 26.62 (0.062) | 30.08 |
| FP8 rowwise + fast acc. | 20.33 (0.120) | 17.71 (0.315) | 25.90 (0.071) | 22.16 (0.148) | 21.53 |
| FP8 rowwise, previous session | 27.23 (0.051) | – | – | 23.48 (0.108) | |
| FP8 tensorwise, previous session | 21.19 (0.111) | – | – | 22.48 (0.134) | |

Distance from the same-session base (fast accumulation, this session), and plain rowwise from the previous session:

| vs base | Fox | Coast | Leaf | Market | Mean |
|---|---|---|---|---|---|
| FP8 rowwise + fast acc. | 20.8 | 18.7 | 25.9 | 22.8 | 22.05 |
| FP8 rowwise | 27.6 | 19.7 | 20.7 | 22.7 | 22.68 |
| FP8 tensorwise | 21.6 | 25.2 | 20.9 | 22.9 | 22.65 |
| 81 f fox, fast acc. | 20.1 | | | | |

**What holds up**
- **Fast accumulation is clearly less accurate than bf16.** It is further from fp32 than eager bf16 on every one of the four videos, by 1.4–10.9 dB, and 5.4 dB on average.
- **The three FP8 variants are indistinguishable by this measure.** Their mean distance from the base is the same within 0.6 dB (22.0–22.7). The per-video values move by up to 7 dB from variant to variant, in both directions.
- **FP8's per-GEMM error is 15× bf16's** (3.7e-2 vs 1.7e-3). Perturbations that large make each video a different sample, with different small details. So a video's PSNR to a reference becomes a draw from a wide distribution: 17.7–27.6 dB here.

**What does not hold up**
- **Plain rowwise's pass last time** was on two videos and by 0.13 dB on the fox. With this spread, that is not evidence that plain rowwise is more accurate than fast accumulation.
- **The per-video bar as an instrument.** It also fails the base on the coast, while the base is 3.1 dB closer to fp32 than eager on average. A rule applied per video, on one seed, breaks down once a change perturbs more than rounding does.

## Activation statistics (`results/g_act_stats.json`)

GEMM output error is relative L2 against fp32 on the bf16 inputs. It is emulated as dequantize-then-matmul in fp32, averaged over blocks and the 6 sampled forwards. The fox is shown; the market is within 2% everywhere.

| Layer | Token spread (max / median token amax) | Channel spread | `tt` | `rr` | `rt` | `tr` | `tt / rr` |
|---|---|---|---|---|---|---|---|
| self-attn q / k / v | 2.5 | 3.4 | 1.97–3.27e-2 | 1.93–3.25e-2 | ≈ `rr` | ≈ `tt` | 1.006–1.017 |
| self-attn out | 2.6 | 4.0 | 3.44e-2 | 3.43e-2 | | | 1.004 |
| cross-attn q | 1.2 | 18.9 | 3.82e-2 | 3.53e-2 | 3.53e-2 | 3.82e-2 | **1.084** |
| cross-attn k / v (text) | **157** | 8.2 | 1.51 / 1.82e-2 | 1.50 / 1.82e-2 | | | 1.00 |
| cross-attn out | 15.8 | 19.3 | 1.55e-2 | 1.53e-2 | | | 1.010 |
| FFN up | 1.6 | 3.6 | 2.09e-2 | 2.07e-2 | | | 1.012 |
| FFN down | 4.8 | 8.6 | 3.21e-2 | 3.20e-2 | | | 1.002 |
| All 300, mean | | | 2.49e-2 | 2.45e-2 | 2.45e-2 | 2.49e-2 | 1.02 |

**Reading the table**
- **Tensorwise vs rowwise.** Per-tensor scales cost 2% more error than per-row scales on average. The worst layer is cross-attention q at 8%, and the worst single (block, layer, step) is 1.27×. The small gap that exists comes from the activation side: `tr` ≈ `tt` and `rt` ≈ `rr`.
- **Token spread is not the cause.** The worst layers have token spread 1.0–1.3. The only large token spread is in cross-attention k/v, and it comes from text padding: the padding tokens have amax 0.0026 and the real tokens 0.48–0.62. It costs nothing there.
- **Subnormals.** Inputs that land below e4m3's smallest normal number make up at most 2% of the values, and only in the text-padding layers.
- **Fox vs market.** The statistics are the same, so they cannot explain a fox-specific loss.
- **Where the hypothesis came from.** It came from treating one video's 6 dB gap as a real effect.

## What this means

- **FP8 linears do not go into the numerically clean base.** Fast accumulation gives 1.17× at 33 f and 1.10× at 81 f. Every video moves further from fp32 than bf16 eager, by 5.4 dB on average. The first run's apparent difference between rowwise and tensorwise, and this run's between plain and fast accumulation, are both inside the spread of single-video PSNR.
- **FP8 linears behave like a trajectory change, not a rounding change.** At this size, they should be judged the way step caching and fewer steps were: the reference-free protocol (4 prompts × 4 seeds, CLIP, blind side-by-side), against `q_steps50`.
- **The accuracy instrument needs replacing for numerical changes.** A lower-noise measure is the error of one transformer forward against an fp32 forward on the same input, sampled along the base trajectory. It does not compound over 50 steps, so it can rank variants that the end-of-trajectory PSNR cannot: rowwise vs fast accumulation vs tensorwise vs Sage FP8-PV vs eager.
- **The ranking at 81 f is unchanged.** Attention is 63% of the forward after FP8 linears, so the lever is still attention.
