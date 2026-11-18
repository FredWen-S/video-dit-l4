# Per-forward error against fp32: predictions (2026-11-18)

Written before running `colab/10_forward_error.ipynb`. Follows `results/L4_fp8_fast_accum.md`.

## Why a new instrument

The FP8 runs showed that one video's PSNR to a reference is a noisy draw once a change is bigger than rounding:
- Three FP8 variants with nearly the same GEMM error swing by up to 7 dB per video, in both directions.
- Their mean distance to the base is 22.0–22.7 dB for all three.
- The base itself misses the per-video bar on the coast.

The 50 sampling steps amplify any perturbation chaotically, so the end-to-end distance mixes the size of a change with the luck of the trajectory. The new instrument removes the trajectory. **Every variant runs one transformer forward on exactly the same inputs, and the output is compared with an fp32 forward on those inputs.**

## The method (`bench/forward_error.py`)

**1. Capture.** One run of the quality-clean base (per-block fullgraph compile, Sage INT8-QK / fp16-PV, 50 steps, CFG 5.0, seed 0) on the 4 prompts at 33 f.
- It keeps the transformer's inputs (latents, timestep, text embeddings) and its output at steps 0, 10, 20, 30, 40 and 49.
- It keeps both branches: cond and uncond. That makes 48 inputs.

**2. One process per variant.** Each runs a forward on all 48 inputs and repeats the first 4.

| Variant | What it is |
|---|---|
| `fp32` | The reference: bf16 weights upcast, TF32 off, memory-efficient SDPA, no compile |
| `eager` | bf16, flash |
| `compile` | Per-block fullgraph compile, flash |
| `compile_emul` | Compile with `emulate_precision_casts=True` |
| `base` | Compile + Sage fp16-PV (the clean base) |
| `sage_fp8pv` | Compile + Sage FP8-PV |
| `fp8_row`, `fp8_row_fast`, `fp8_tensor` | Base + W8A8 FP8 linears: rowwise, rowwise with fast accumulation, tensorwise |

**3. The metrics.** For each input:
- **Branch error:** rel L2 of the velocity output against fp32.
- **Low-pass error:** the same after 4×4 spatial average pooling of the latent.
- **Guided error:** the error of `u + 5(c − u)`, which is what the sampler uses.

**4. 81 f.** The fox only, at steps 0, 20 and 49. Variants: `fp32`, `eager`, `base`, `sage_fp8pv`, `fp8_row_fast`.

**Known end-to-end facts the instrument should agree with:**

| vs fp32 (dB) | Fox | Market |
|---|---|---|
| Eager | 28.1 | 23.8 |
| Compile | 35.6 | 26.2 |
| Compile, emulated casts | 27.8 | 23.9 |
| Base | 35.4 | 26.6 |
| Sage FP8-PV | 32.0 | 23.2 |

- FP8 linears average 21.5 dB on four videos, against eager's 27.0.
- The microbenchmarks put the per-call errors at:
  - flash attention 2.3e-3, Sage fp16-PV 1.2e-2, Sage FP8-PV 3.9e-2;
  - a bf16 GEMM 1.7e-3, an FP8 GEMM 3.7e-2.

## Rough size

**Per block:**
- Eager rounds the residual stream and the adaLN outputs to bf16 several times. That is about 2.3e-3 rms each.
- Compile rounds them about once.
- Sage fp16-PV adds about 1.2e-2 on the attention output. That output is maybe 20% of the residual's size, so it adds 2–3e-3.
- An FP8 GEMM adds 3.7e-2 on every linear output.

**Over 30 blocks,** if the errors add like independent noise, the branch output ends up at roughly 1% for eager. The base lands near eager, Sage FP8-PV about 2× higher, and FP8 linears 3–10× higher.

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| H1 | **The instrument reproduces.** `base` on the captured inputs reproduces the pipeline's own outputs bit for bit on all 48. Every variant gives bit-identical outputs when a forward is repeated | any differs |
| H2 | **Size:** eager's branch error, averaged over the 48 inputs, is 3e-3 to 3e-2 | outside |
| H3 | **Cast elision is visible in one forward:** `compile` / `eager` is 0.3–0.8, and `compile_emul` / `eager` is 0.8–1.25 | either outside |
| H4 | **Sage fp16-PV:** `base` / `compile` is 1.0–2.0, and `base` / `eager` is 0.5–1.0 | either outside |
| H5 | **Sage FP8-PV:** `sage_fp8pv` / `base` is 1.5–4 | outside |
| H6 | **FP8 linears are the largest error:** `fp8_row_fast` / `base` is 3–10, and above `sage_fp8pv` | outside, or below `sage_fp8pv` |
| H7 | **The three FP8 variants are equal within 10%** (mean branch error): tensorwise highest, and fast accumulation within 3% of plain rowwise | spread above 10%, tensorwise not highest, or fast vs plain differs by more than 3% |
| H8 | **Stable ranking:** per-prompt means of each variant vary by at most 1.3× (max/min over the 4 prompts). Every pair of variants whose overall means differ by more than 1.3× is ordered the same way on all 4 prompts and all 6 steps | any variant above 1.3×, or any such pair flips on a prompt or a step |
| H9 | **CFG amplifies the error:** guided error / branch error is 1.5–5 for every variant (5c − 4u adds the branch errors with weights 5 and 4, but they are partly correlated) | outside for any variant |
| H10 | **81 f:** each variant's branch error on the fox at steps 0, 20 and 49 is within ±25% of its 33 f value at the same steps. The microbenchmarks show no per-call error growth with length | any outside |

## What each outcome means

- **H3–H6 hold.** The single-forward error ranks changes the same way as the end-to-end averages, without the trajectory noise. It becomes the numeric check:
  - A change no larger than eager's error is rounding-level and can use the fp32 bar.
  - A change above it is a trajectory change and needs the reference-free protocol.
- **H3 fails.** The cast elision that moved the end-to-end result 7.5 dB on the fox does not show up in one forward. The end-to-end gap then comes from how the error builds up over the steps, not from its size per forward: for example, a bias that adds up rather than noise that averages out. The low-pass and guided errors are the first place to look.
- **H4 fails high** (base above eager). One forward ranks the base worse than eager, but end to end it is closer to fp32 on 3 of 4 videos. Per-forward rel L2 then does not predict the end-to-end effect, and the instrument needs a different weighting, such as the low-pass error.
- **H8 fails.** One forward is noisy too, and the instrument needs more inputs.
