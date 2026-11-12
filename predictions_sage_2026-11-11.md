# 8-bit attention (SageAttention2) on top of compile: predictions before the run (2026-11-11)

Written before running `colab/05_sage_attention.ipynb`.

## The change

Attention in the transformer moves from flash attention (bf16) to SageAttention 2.2.0. It is selected through diffusers' `set_attention_backend("sage")` and built from source for sm_89. Everything else stays as it is: per-block compile, 50 steps, CFG 5.0, seed 0, 480×832.

On sm_89, `sageattn` dispatches to `sageattn_qk_int8_pv_fp8_cuda`:
- **Q and K:** quantized to INT8, per thread. K is mean-smoothed first.
- **QKᵀ:** INT8 tensor cores.
- **P and V:** FP8 (e4m3); PV accumulates in "fp32+fp16".

The L4 runs INT8 and FP8 tensor cores at twice its bf16 rate, so the ideal speedup of the attention kernel is 2×. Quantizing Q, K and V adds a memory-bound pass of about 200 MB per self-attention call at 33 frames (about 1 ms at 231 GB/s, about 30 ms per forward). It scales with N, while attention scales with N², so the overhead matters less at 81 frames.

A second variant isolates the PV precision: `_sage_qk_int8_pv_fp16_cuda`, with INT8 QK and fp16 PV accumulating in fp32. Only QKᵀ runs at the higher rate, so the ideal speedup is about 1.33×.

## Starting point (compile, flash)

| | 33 f (torch 2.14) | 81 f (torch 2.11) |
|---|---|---|
| Mean forward | 1,474 ms | 5,286 ms |
| Flash attention | 714 ms (48%) | 3,536 ms (67%) |
| GEMM | 591 ms | 1,413 ms |
| Everything else | 155 ms | 421 ms |
| Ceiling if attention is 2× faster | 1.32× | 1.50× |

The quality references come from `results/L4_compile_diagnosis.md`. These are PSNR distances to the fp32 transformer:

| | Fox | Market |
|---|---|---|
| Compile | 35.6 dB | 26.2 dB |
| Eager bf16 | 28.1 dB | 23.8 dB |
| Bar for "no worse than the bf16 baseline" (eager minus 1 dB) | 27.1 dB | 22.8 dB |

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| S1 | If torch is still 2.14.1, compile reproduces `compile_rerun_33f` and fp32 reproduces `fp32_33f` bit for bit (`frames_sha256.json`) | any hash differs while torch is 2.14.1 |
| S2 | Microbenchmark (B=1, H=12, D=128, self-attention). Speed: Sage, including quantization, is 1.4–2.0× faster than flash at N = 14,040 and 1.5–2.2× at N = 32,760. Accuracy: its relative L2 error against an fp32 reference is 3–20× flash-bf16's. A per-channel bias on K raises Sage's error by less than 2×, because K-smoothing removes it. | any of these outside its range |
| S3 | End to end, Sage on top of compile: mean forward 1.14–1.30× faster at 33 f and 1.25–1.50× at 81 f | either outside its range |
| S4 | Attention plus quantization kernels per forward: 33 f 714 → 375–510 ms, 81 f 3,536 → 1,770–2,360 ms. Mean SM clock within ±4% of compile. | any outside its range |
| S5 | Sage vs compile at 33 f: mean PSNR 20–30 dB and LPIPS ≤ 0.12 on every video. Same scene with changed details, like compile vs eager; not a different sample. | any video outside |
| S6 | Sage vs fp32 meets the bf16-baseline bar: fox ≥ 27.1 dB and market ≥ 22.8 dB | either below |
| S7 | The fp16-PV variant is closer to compile than the FP8-PV variant on ≥ 3 of 4 videos, and slower: attention 1.1–1.4× faster than flash at 33 f | fewer than 3, or outside 1.1–1.4× |
| S8 | 81 f fox, Sage vs compile: mean PSNR 17–28 dB | outside |

## What each outcome means

- **S3 and S6 hold.** 8-bit attention is a real speedup that costs no more accuracy than bf16 eager already does. It becomes part of the baseline, and 81 frames is where it pays.
- **S6 fails but S7's fp16-PV variant passes.** The FP8 quantization of P·V is where accuracy goes. The fp16-PV variant is then the right trade-off, if its speedup is worth having.
- **S2 holds but S3 falls short.** The power cap or quantization overhead takes the gain. The clock data in S4 shows which.
