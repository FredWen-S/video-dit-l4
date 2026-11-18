# W8A8 FP8 linears on top of compile + Sage fp16-PV, L4 (2026-11-18)

Predictions: `predictions_fp8_2026-11-18.md`. Notebook: `colab/08_fp8_linears.ipynb`. Code: `bench/fp8_linear.py`, `bench/fp8_micro.py`.

**Result: 3 of 9 predictions held.** The two scale granularities split speed and quality between them:

| Variant | Speed | Quality vs fp32 |
|---|---|---|
| **Rowwise** | Meets the fp32 bar on both reference videos, but makes the forward **1.38× slower** (0.72× at 33 f, 0.79× at 81 f) | Meets the bar on both |
| **Tensorwise** | **1.12× faster** at 33 f | Fails the bar on both, by 5.9 dB on the fox |

**Why rowwise is slow.** PyTorch's rowwise `_scaled_mm` on sm_89 is a CUTLASS 2.x kernel. With fp32 accumulation (the default here), it runs at 31–38 TFLOPS, half of bf16. With fast accumulation, it runs at 90–113 TFLOPS. In the microbenchmark, fast accumulation gave the same error to three digits at these K. So rowwise with fast accumulation should give tensorwise's speed with rowwise's quality. That was measured only in the microbenchmark, not end to end; it is the next run.

**Memory.** Peak memory drops by 1.1–1.3 GB, because the block weights are stored in FP8.

## Setup

- Same environment as the earlier runs:
  - torch 2.14.1+cu130, diffusers 0.42.1, transformers 5.22.2, SageAttention 2.2.0, Triton 3.8.0;
  - driver 580.82.07, nvcc 13.0.
  - Driver, nvcc and Triton are logged for the first time.
- The base is per-block fullgraph compile plus Sage INT8-QK / fp16-PV, at 50 steps, CFG 5.0, seed 0. All runs were in one session.
- **FP8:** `fp8_linear.install` replaced 300 linears, 10 per block:
  - self-attention q/k/v/o;
  - cross-attention q/k/v/o;
  - FFN up and down.
- **Quantization:**
  - weights are e4m3, quantized once;
  - activations are quantized in each forward;
  - fp32 accumulation, bf16 output.
- **Fused quantization:** Inductor fused each activation's quantization (amax, scale, cast) into the kernel that produces it. The traces show kernels named `triton_red_fused__scaled_mm__to_copy_abs_add_amax_clamp_div_mul_native_layer_norm...` and `..._gelu...`.

## Verdicts

| # | Prediction | Measured | |
|---|---|---|---|
| F1 | rowwise `_scaled_mm` works on sm_89 | works (CUTLASS `enable_2x_kernel_for_sm89` with a rowwise-scale epilogue) | ✓ |
| F2 | GEMM only, M = 14,040: FP8 1.5–1.9× bf16 on all shapes; rowwise within 10% of tensorwise; fast accumulation ≤ 15% faster | tensorwise 1.73 / **1.43** / 1.90×; rowwise **0.52 / 0.55 / 0.63×**; fast accumulation **2.9× faster** than without | ✗ |
| F3 | error: bf16 ≤ 5e-3; FP8 2–6e-2; outlier channels raise tensorwise error < 1.5× | bf16 1.66e-3; FP8 3.4–3.75e-2; tensorwise with outliers ×1.00 | ✓ |
| F4 | rowwise end to end: 1.09–1.19× at 33 f, 1.06–1.13× at 81 f | **0.72×** and **0.79×** | ✗ |
| F5 | 33 f rowwise: GEMM 600 → 300–400 ms; quantization 40–120 ms; clock within ±5% | GEMM 599 → **1,111 ms**; quantization ≈ 94 ms; clock **+16%** | ✗ |
| F6 | rowwise vs fp32: fox meets the bar, market does not | fox 27.23 (bar 27.1), market **23.48** (bar 22.8): both meet it | ✗ |
| F7 | tensorwise within 2 dB of rowwise on both videos | fox **6.0 dB** worse, market 1.0 dB | ✗ |
| F8 | rowwise vs base, 22–30 dB on each video | 27.6 / **19.7 / 20.7** / 22.7 dB | ✗ |
| F9 | fp32 reproduces `fp32_33f`; the base reproduces `pv16_33f` or `q_steps50` seed 0 | fp32 2 of 2 identical; base 4 of 4 identical to `pv16_33f`, 0 of 4 to `q_steps50` seed 0 | ✓ |

F6 failed in the good direction: rowwise is closer to fp32 than I predicted. F2, F4 and F5 all fail for the same reason, the slow rowwise kernel. I assumed every FP8 path on sm_89 runs at FP8 tensor-core rate. Only cuBLASLt's tensorwise path does, and CUTLASS's rowwise path does only with fast accumulation.

## Microbenchmark (`results/fp8_micro.json`)

GEMM only, on pre-quantized inputs, with Gaussian activations. Times in ms; the speedup vs bf16 is in parentheses.

| M | Shape | bf16 | FP8 tensorwise | FP8 rowwise | FP8 rowwise + fast acc. | Rowwise + quantization (compiled) |
|---|---|---|---|---|---|---|
| 14,040 | qkvo 1536→1536 | 0.90 | 0.52 (1.73×) | 1.74 (0.52×) | 0.61 (1.48×) | 1.96 |
| 14,040 | FFN up 1536→8960 | 6.49 | 4.53 (1.43×) | 11.77 (0.55×) | 4.11 (1.58×) | 12.13 |
| 14,040 | FFN down 8960→1536 | 6.81 | 3.58 (1.90×) | 10.82 (0.63×) | 3.42 (1.99×) | 11.99 |
| 32,760 | qkvo | 2.46 | 1.50 (1.64×) | 5.05 (0.49×) | 1.49 (1.66×) | 5.13 |
| 32,760 | FFN up | 15.37 | 12.70 (1.21×) | 27.25 (0.56×) | 10.04 (1.53×) | 28.47 |
| 32,760 | FFN down | 16.95 | 9.63 (1.76×) | 24.65 (0.69×) | 8.39 (2.02×) | 28.47 |

**Error** (relative L2 against an fp32 matmul of the same bf16 inputs):

| Variant | Gaussian | 8 outlier channels at 30× |
|---|---|---|
| bf16 | 1.66e-3 | 1.66e-3 |
| FP8 tensorwise | 3.75e-2 | 3.74e-2 |
| FP8 rowwise | 3.74e-2 | 3.39–3.56e-2 |
| FP8 rowwise + fast accumulation | same as rowwise to three digits | same as rowwise to three digits |

**The FP8 rate on the L4.** FP8 peaks at about 127 TFLOPS here (tensorwise, qkvo). That is far below the datasheet's dense FP8 rate, and the cause is the 72 W power cap. FFN up is the weakest shape: 1.21–1.43× tensorwise.

**Why fast accumulation matters so much.** Fast accumulation is what lets the CUTLASS kernel use the FP8 MMA at full rate. Without it, the kernel runs at about half of bf16, which points to extra work in each k-iteration. My guess is a periodic promotion of the accumulator into fp32. I did not check CUTLASS's sm_89 FP8 path to confirm this.

**Model vs microbenchmark.** The micro model predicts the end-to-end GEMM time:
- Base: 18.7 ms per block, or 561 ms per forward, against 599 ms measured.
- Rowwise: 33.0 ms per block, or 990 ms per forward, against 1,111 ms measured. The microbenchmark leaves out the cross-attention k/v on 512 text tokens.

## Speed

Mean forward over the 100 forwards of each video (4 videos at 33 f, 1 at 81 f).

| | Mean forward | vs base | SM clock | Peak memory | Wall per video |
|---|---|---|---|---|---|
| 33 f base | 1,358 ms | 1.00× | 1,110 MHz | 3.82 GB | 147.0 s |
| 33 f FP8 rowwise | 1,877 ms | **0.72×** | 1,289 MHz | 2.55 GB | 199.0 s |
| 33 f FP8 tensorwise | 1,217 ms | **1.12×** | 1,106 MHz | 2.55 GB | 133.0 s |
| 81 f base | 4,449 ms | 1.00× | 1,173 MHz | 4.35 GB | 472.3 s |
| 81 f FP8 rowwise | 5,663 ms | **0.79×** | 1,296 MHz | 3.25 GB | 593.8 s |

The base's mean forward matches the 2026-11-13 session: 1,358 vs 1,365 ms at 33 f, and 4,449 vs 4,423 ms at 81 f.

**Kernel time per forward**, in ms, from one profiled forward. Tensorwise ran without a trace.

| | GEMM | Attention | Attention quantization | Norm (Inductor) | Other | Total |
|---|---|---|---|---|---|---|
| 33 f base | 599 | 544 | 22 | 80 | 103 | 1,347 |
| 33 f rowwise | 1,111 | 451 | 21 | 118 | 159 | 1,859 |
| 81 f base | 1,346 | 2,627 | 57 | 200 | 250 | 4,479 |
| 81 f rowwise | 2,617 | 2,343 | 56 | 292 | 379 | 5,687 |

**The power cap at work.** The slow rowwise GEMM draws less power per unit of time, so the SM clock rose 16% at 33 f and 10% at 81 f. Sage's attention kernel, which was not changed, got faster by the same amount: 544 → 451 ms and 2,627 → 2,343 ms. A kernel's time on this GPU depends on what runs next to it.

**Cost of the activation quantization.** It lives inside the fused norm and GELU kernels, so it shows up as the growth of "norm" plus "other": 94 ms at 33 f and 221 ms at 81 f. That is within the predicted 40–120 ms at 33 f. Those kernels ran at a 16% higher clock, so at the base clock it would be a little more.

**Accounting for tensorwise** (no trace, so this is inferred):
- The microbenchmark puts its GEMM at about 11.2 ms per block, about 260 ms per forward less than the base.
- The fused quantization costs about 94 ms.
- That nets about 170 ms. The measured gain was 141 ms.

## Quality

PSNR is the mean over frames, computed from npy. LPIPS is in parentheses. The fp32 reference exists for the fox (video 0) and the market (video 3), and it was rerun in this session, bit-identical to `fp32_33f`.

| vs fp32 | Fox (bar ≥ 27.1) | Market (bar ≥ 22.8) |
|---|---|---|
| Base (compile + Sage fp16-PV) | 35.39 (0.007) | 26.62 (0.062) |
| FP8 rowwise | **27.23** (0.051) | **23.48** (0.108) |
| FP8 tensorwise | 21.19 (0.111) ✗ | 22.48 (0.134) ✗ |

| vs base | Video 0 | Video 1 | Video 2 | Video 3 |
|---|---|---|---|---|
| 33 f rowwise | 27.6 (0.053) | 19.7 (0.245) | 20.7 (0.229) | 22.7 (0.129) |
| 33 f tensorwise | 21.6 (0.106) | 25.2 (0.108) | 20.9 (0.210) | 22.9 (0.118) |
| 81 f rowwise (fox) | 24.2 (0.146) | | | |

**Rowwise.**
- It clears the bar on the fox by 0.13 dB and on the market by 0.68 dB. Both margins are thin.
- It gives up 8.2 dB (fox) and 3.1 dB (market) of the base's distance to fp32.
- Videos 1 and 2 have no fp32 reference, and against the base they moved the most of any video (19.7 and 20.7 dB). The bar is therefore judged on half the videos, and the half without a reference moved more. An fp32 reference for videos 1 and 2 is needed before rowwise is called clean.

**Tensorwise vs rowwise.**
- The microbenchmark said tensorwise and rowwise have the same error, outlier channels included. In the model, tensorwise is 6 dB worse on the fox.
- So real activations have a structure that the synthetic test lacked. Channel outliers don't explain it, since e4m3 handled them.
- The likely candidate is a spread across tokens. A single tensor scale set by a few very large tokens pushes the ordinary tokens' values down toward e4m3's subnormal range, where they keep fewer mantissa bits. Per-token scales avoid that.
- This is a hypothesis. Logging each linear's global amax against its per-row amax distribution would test it.

## Reproducibility (F9 and two loose ends)

**The 33 f base reproduced `pv16_33f` (2026-11-13) bit for bit, on all 4 videos.** The cause of last time's mismatch is therefore not the environment, and not the compiled Sage path across sessions.

Two mismatches remain. The zip has no npy, so their sizes below are from the mp4s (ffmpeg PSNR filter). The two encodings of identical npy give infinite PSNR, so these numbers are a rough size, not a metric:

| Pair | mp4 PSNR |
|---|---|
| `q_steps50` seed 0 (2026-11-14) vs `pv16_33f`, fox | 37.5 dB |
| `q_steps50` seed 0 (2026-11-14) vs `pv16_33f`, market | 32.2 dB |
| 81 f base now vs `pv16_81f` (2026-11-13) | 33.7 dB |

**`q_steps50` seed 0.**
- It does not match `pv16_33f`, and neither does today's base match it. The difference is about the size of rounding noise, not a different trajectory.
- Its config matches today's apart from `--seeds 0 1 2 3`, and seed 0's four videos ran first, in the same order.
- The q_steps50 run itself is the odd one out. The cause is still unknown.

**81 f base vs `pv16_81f`.**
- The 81 f base also failed to reproduce `pv16_81f`, at the same rounding-noise size.
- The 33 f base, run with the same code and flags, did reproduce.
- One difference is that this 81 f run profiled a forward (`--kernels`) and `pv16_81f` did not. At 33 f, the profiled run still matched, so profiling alone is not a sufficient explanation.

**Status.** Both are rounding-level differences, but they mean "bit-identical across sessions" holds at 33 f and not reliably at 81 f.

## What this means

- **Tensorwise:** 1.12× faster at 33 f but fails the fp32 bar. It is not kept.
- **Rowwise with fp32 accumulation:** meets the bar on the two reference videos but is 1.4× slower. It is not kept.
- **Next run: rowwise with fast accumulation** (`--fp8-fast-accum`, one flag, one variable).
  - The microbenchmark gives it tensorwise's GEMM speed and rowwise's error.
  - Expected: about 1.10× at 33 f, with quality close to rowwise.
  - At 81 f, GEMM is only 30% of the forward, so the ceiling is lower. The clock effect above also cuts both ways: a faster GEMM draws more power and may slow attention.
  - Before calling it clean, add fp32 references for videos 1 and 2.
- **Ranking at 81 f.** FP8 linears are worth at most about 1.1×, against 1.35× for FP8 attention. Attention is still the lever at 81 frames.
