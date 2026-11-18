# FP8 rowwise with fast accumulation, plus fp32 references for all four videos: predictions (2026-11-18)

Written before running `colab/09_fp8_fast_accum.ipynb`. Follows `results/L4_fp8_linears.md`.

## The change

The first FP8 run gave two results:
- Rowwise met the fp32 bar on the fox and the market, but at 0.72× speed.
- The microbenchmark put rowwise with `use_fast_accum=True` at tensorwise's GEMM speed, with rowwise's error.

**One variable** goes on top of `--fp8-linears rowwise`: `--fp8-fast-accum`. Everything else is the quality-clean base: per-block fullgraph compile, Sage INT8-QK / fp16-PV, 50 steps, CFG 5.0, seed 0.

**Two additions to the quality check:**
- **fp32 references for videos 1 (coast) and 2 (leaf).** In the last run, these two moved furthest from the base (19.7 / 20.7 dB), but the fp32 bar was judged only on videos 0 and 3. The fp32 run covers all four prompts, because the npy of the earlier references are not on Colab.
- **Bars for the two new videos:** eager bf16's distance to fp32, minus 1 dB, the same rule as for the fox and the market. Eager runs on prompts 1 and 2 in this session. The fox and market keep their fixed bars, 27.1 and 22.8 dB.

**One diagnostic, no variable changed: `bench/fp8_act_stats.py`.** Tensorwise lost 6 dB on the fox even though the microbenchmark showed no error difference. The script hooks every block linear in an eager run of the fox and the market and measures:
- activation spread per token and per channel;
- the share of inputs that fall below e4m3's smallest normal number after scaling;
- the GEMM output error of four combinations of activation scale × weight scale (tensor or row), emulated in fp32.

**Reproducibility checks:**
- base 33 f vs `f_base_33f` and `pv16_33f`;
- eager vs `flash_33f`;
- fp32 vs `fp32_33f`;
- base 81 f vs `f_base_81f`, which used the same flags including `--kernels`, and vs `pv16_81f`.

## Expected numbers, from the microbenchmark

**GEMM time per block, 33 f (M = 14,040):**

| | qkvo ×6 | FFN up | FFN down | Per block | Per forward |
|---|---|---|---|---|---|
| bf16 | 0.90 ms each | 6.49 | 6.81 | 18.7 ms | 561 ms (599 measured) |
| rowwise + fast accumulation | 0.61 ms each | 4.11 | 3.42 | 11.2 ms | 335 ms |

**Net gain at 33 f:**
- GEMM saves about 226 ms per forward.
- The fused activation quantization costs about 94 ms (measured in the rowwise run).
- That nets about 130 ms: 1,358 → about 1,225 ms, or 1.11×.

**Net gain at 81 f:**
- GEMM saves about 590 ms.
- Quantization costs about 220 ms.
- That nets about 370 ms: 4,449 → about 4,080 ms, or 1.09×.

**The power cap:**
- Tensorwise, with similar GEMM times, ran at the base's clock, so I don't expect a clock effect at 33 f.
- At 81 f, a faster GEMM draws more power and may cost a little attention speed. The rowwise run showed the reverse effect.

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| G1 | **Reproducibility at 33 f:** base 4 of 4 identical to `f_base_33f` and `pv16_33f`; eager (prompts 1, 2) identical to `flash_33f`; fp32 (prompts 0, 3) identical to `fp32_33f` | any differs |
| G2 | **81 f base vs `f_base_81f`** (same flags, including `--kernels`): identical. It differs from `pv16_81f` as last time | identical to `pv16_81f`, or differs from `f_base_81f` |
| G3 | **33 f speed:** mean forward 1.06–1.15× faster than the same-session base | outside |
| G4 | **33 f kernel split:** GEMM 599 → 320–420 ms; norm + other grows 60–120 ms; SM clock within ±5% of the base | any outside |
| G5 | **81 f speed:** 1.04–1.12× faster | outside |
| G6 | **Fast accumulation does not change quality:** vs fp32, fox 26.7–27.7 dB and market 23.0–24.0 dB, i.e. within 0.5 dB of plain rowwise (27.23 / 23.48) | either outside |
| G7 | **Eager bf16 vs fp32:** coast and leaf both 24–30 dB (fox 28.1, market 23.8). Compile vs eager was 27.2 / 29.2 dB on these two | either outside |
| G8 | **The bar on all four videos:** fast-accumulation rowwise meets it on 0 and 3, and misses it on 1 and 2 by at least 2 dB. Rowwise moved these two 19.7 / 20.7 dB from the base, much further than eager differs from compile. So rowwise FP8 does not go into the numeric baseline | meets the bar on 1 or 2, or misses it on 0 or 3 |
| G9 | **Diagnostic, fox:** tensorwise's loss comes from the activations' single scale, not the weights'. Averaged over layers, `tr` (activation tensor, weight row) is within 20% of `tt`, and `rt` within 20% of `rr`. The ten worst (block, layer) pairs by `tt / rr` have token spread (global amax / median token amax) above 30 | `rt` closer to `tt` than `tr` is, or the worst pairs have token spread below 30 |

## What each outcome means

- **G3 and G6 hold, G8 as predicted.** Rowwise FP8 with fast accumulation is a 1.1× option at 33 f that costs more accuracy than bf16 eager on half the prompts. It stays an option, not part of the clean base. Attention remains the lever at 81 f.
- **G8 fails in the good direction** (meets the bar on all four). FP8 linears go into the numeric baseline, and the next step is stacking them with Sage FP8-PV.
- **G9 holds.** A fix for tensorwise would be smoothing (SmoothQuant-style), or keeping the few high-spread layers in bf16. That is a later experiment.
- **G2 fails.** The 81 f mismatch is run-to-run nondeterminism at 81 f, not the profiler. Then 81 f comparisons across sessions need PSNR, not hashes.
