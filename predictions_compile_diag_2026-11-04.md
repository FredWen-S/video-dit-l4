# What does compile change? Predictions before the run (2026-11-04)

Written before running `colab/04_compile_diagnosis.ipynb`.

## Why

The noise-floor run (`results/L4_noise_floor.md`) gave three reference points:
- An exact attention-kernel swap moves the output to 36 dB from eager at 33 frames and 39 dB at 81.
- A different seed moves it to 10–13 dB.
- Compile lands at 25–30 dB at 33 frames and 21.6 dB at 81.

Two features of the compile divergence point to a systematic change rather than random rounding: it depends strongly on the prompt (4.1 dB spread, against 0.4 dB for the kernel swap), and it grows with video length (the kernel swap does not).

There are two open questions:
1. **What causes it?** The leading suspect is *cast elision*. Eager rounds each op's result to bf16 before the next op reads it. When Inductor fuses two ops, it drops that downcast–upcast pair and keeps the value in fp32. Setting `torch._inductor.config.emulate_precision_casts = True` keeps the pairs.
2. **Is compile worse or better?** Dropping roundings could make compile *closer* to full precision than eager. Distance to eager cannot tell; a full-precision reference can.

## Runs

33 frames, 50 steps, CFG 5.0, seed 0, all in one Colab session.

| Run | What it is | Prompts |
|---|---|---|
| R `flash_33f` | eager bf16, flash attention (the baseline path) | all 4 |
| C `compile_33f` | compile per block, default Inductor settings (rerun) | all 4 |
| CE `compile_emul_33f` | as C, plus `emulate_precision_casts=True` | all 4 |
| F `fp32_33f` | eager, transformer in fp32 with TF32 off, memory-efficient attention (flash has no fp32 kernel) | fox (0) and night market (3) |

About F:
- Its weights are the same bf16 weights upcast to fp32, so it differs from R only in the precision of activations and math. That is the axis compile moves along.
- The two prompts are compile's closest and farthest cases.
- Text embeddings, the VAE (fp32) and the sampler are identical.

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| D1 | R reproduces the earlier eager frames bit for bit | any hash differs |
| D2 | Cast elision explains the divergence: CE vs R mean PSNR ≥ 32 dB on every video, within 4 dB of the 36 dB floor | any video below 32 dB |
| D3 | CE costs 0–5% over C per forward. The extra roundings happen in registers, so memory traffic is unchanged. | outside 0–5% |
| D4 | R vs F: mean PSNR 22–34 dB on both videos. bf16 activations perturb every op, which is more than the attention-only swap (36 dB) but random rather than biased. | outside 22–34 dB |
| D5 | C is closer to F than R is, on both videos: dropping roundings makes compile the more accurate bf16 path | R closer on either video |
| D6 | The fp32 forward is 3–6× slower than eager bf16 (6–12 s per forward at 33 f) | outside 3–6× |
| D7 | CLIP score (now fixed): \|efficient − flash\| < \|seed 1 − flash\| on ≥ 3 of 4 prompts (this is N7, run again). Also, \|C − R\| is within the seed spread \|seed 1 − flash\| on ≥ 3 of 4 prompts. | either count below 3 |

## What each outcome means

- **D2 and D5 both hold.** The divergence is cast elision, and it moves compile toward full precision. Compile is then a better approximation of the model than eager, not a worse one. The 81-frame fox defect becomes a property of that sample, not a sign that compile degrades quality.
- **D2 holds and D5 fails.** Cast elision is the cause, but eager's extra roundings happen to sit closer to fp32 here. With only two videos, call it unresolved rather than "compile is worse".
- **D2 fails.** Something else in Inductor changes the numerics. Candidates:
  - bias adds split out of the cuBLAS epilogue and rounded twice;
  - a different reduction order in LayerNorm and RMSNorm;
  - a different GELU-tanh implementation.
  
  The next step would be to compile one sub-module at a time and compare.
