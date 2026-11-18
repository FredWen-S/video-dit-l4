# W8A8 FP8 linears on top of compile + Sage fp16-PV: predictions before the run (2026-11-18)

Written before running `colab/08_fp8_linears.ipynb`.

## The change

Every `nn.Linear` inside the 30 transformer blocks becomes an FP8 (e4m3) GEMM via `torch._scaled_mm` (`bench/fp8_linear.py`, `--fp8-linears`):
- **Which linears:** self- and cross-attention q/k/v/o, and FFN up and down.
- **Weights:** quantized once.
- **Activations:** quantized in each forward. Compile fuses the amax, scale and cast into Triton kernels.
- **Granularity:** two variants. **rowwise** has one scale per token and one per output channel. **tensorwise** has one scale per tensor.
- **Accumulation:** fp32 by default. cuBLASLt's faster "fast accumulation" is measured only in the microbenchmark.

Everything else stays as in the quality-clean base: per-block fullgraph compile, Sage INT8 QK / fp16 PV, 50 steps, CFG 5.0, seed 0. The 50-step trajectory is unchanged, so this is a numerical change, and the fp32 bar applies again: fox ≥ 27.1 dB, market ≥ 22.8 dB.

Not torchao: no torchao release lists torch 2.14. On sm_89, its float8 path ends in `torch._scaled_mm` anyway; its faster kernels need sm_90.

## Starting point (compile + Sage fp16-PV)

| | 33 f | 81 f |
|---|---|---|
| Mean forward | 1,365 ms | 4,423 ms |
| GEMM per forward | 600 ms (44%) | ~1,400 ms (32%), estimated from compile at 81 f |
| Ceiling if every GEMM is exactly 2× faster | 1.28× | 1.19× |

**Overhead of quantizing the activations.** Each FP8 linear quantizes its own input: one bf16 read and one fp8 write, after an amax pass. Per block that is:
- seven 14,040 × 1,536 inputs (q, k, v, o, cross q, cross o, FFN up), about 0.3 ms each at 230 GB/s;
- one 14,040 × 8,960 input (FFN down), about 1.6 ms.

That totals about 3.6 ms per block, or 110 ms per forward at 33 f, if nothing fuses with the preceding kernel. With some fusion it could be 50 ms.

**Why FP8 should tolerate outlier channels.** e4m3 is a floating-point format: each value keeps about 3 mantissa bits of relative precision over a wide range. A channel 30× larger than the rest moves the scale but leaves the small values with nearly the same relative precision. Integer formats don't behave this way. So per-tensor scaling should cost little more than per-row.

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| F1 | `_scaled_mm` accepts rowwise scales on sm_89 (L4) with torch 2.14 | the microbenchmark records it as unsupported |
| F2 | Microbenchmark, GEMM only, M = 14,040: FP8 is 1.5–1.9× faster than bf16 on all three shapes. Rowwise and tensorwise are within 10% of each other. Fast accumulation is at most 15% faster than without | any outside |
| F3 | Microbenchmark error, Gaussian inputs: bf16 ≤ 5e-3 relative L2; FP8 (either granularity) 2e-2 to 6e-2. With 8 outlier channels at 30×, tensorwise error rises by less than 1.5× | any outside |
| F4 | End to end, rowwise, mean forward vs the uncached base in the same session: 33 f 1.09–1.19× faster, 81 f 1.06–1.13× | either outside |
| F5 | Kernel split at 33 f, rowwise: GEMM 600 → 300–400 ms per forward; new quantization kernels 40–120 ms; mean SM clock within ±5% of the base | any outside |
| F6 | Against fp32, rowwise: the fox meets the bar (≥ 27.1 dB), the market does not (< 22.8 dB). Sage FP8-PV, one FP8 product per block, already left the market only 0.4 dB above the bar; this is seven FP8 GEMMs per block | either the other way |
| F7 | Tensorwise vs rowwise: their distances to fp32 are within 2 dB of each other on both videos | more than 2 dB apart on either |
| F8 | Rowwise vs the base (no FP8): PSNR 22–30 dB on each of the 4 videos | any outside |
| F9 | Reproducibility: fp32 reproduces `fp32_33f` bit for bit. The bf16 base reproduces either `pv16_33f` (session of 2026-11-13) or seed 0 of `q_steps50` (2026-11-14). This tells whether last time's mismatch came from the compiled/Sage path or from everything | fp32 differs, or the base matches neither |

## What each outcome means

- **F4 and F6 both hold for the fox only.** FP8 linears give about 1.1× and cost about as much accuracy as FP8 attention did. That is too little speed for what it costs at 81 frames, where attention dominates. It would be kept only as an option.
- **F6 holds on both videos.** FP8 linears go into the numeric baseline.
- **F2 holds but F4 falls short.** The activation quantization passes or the power cap take the gain. F5 shows which. The fix would be fusing the quantization into the preceding norm or activation kernels.
