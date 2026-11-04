# Baseline on Colab L4 (2026-11-03)

Wan2.1-T2V-1.3B, 480×832, 50 UniPC steps, CFG 5.0, bf16 transformer, fp32 VAE, PyTorch SDPA (flash). Software: torch 2.11.0+cu130, diffusers 0.42.1, transformers 5.22.1. Raw data: `base_33f/report.json` and `base_81f/report.json`. Predictions: `../predictions_baseline_2026-11-03.md`.

The four 33-frame runs differ by at most 0.6 s (0.3%), and the two 81-frame runs by 0.5 s. Timing does not depend on the prompt, as expected with fixed shapes. All six videos look correct.

## First attempt: out of memory

The first runs failed in the VAE decode at every length, including the 17-frame smoke test.

- The L4 exposes 22.03 GiB, not 24 GB.
- With the umT5 encoder resident, the pipeline holds 14.7 GB of weights.
- The fp32 VAE decode then needs another ~8.8 GB of activations (measured peak 12.2 GB on top of 3.4 GB of weights).

Fix (commit 84f85b1): encode all prompts first, then move the text encoder to the CPU and drop it from the pipeline. This does not change the outputs. Prompt embeddings are computed with the same 512-token length the pipeline uses.

## Measurements

| | 33 frames (N = 14,040) | 81 frames (N = 32,760) |
|---|---|---|
| Wall time per video | 214.2 s (3.57 min) | 678.4 s (11.31 min) |
| Transformer (100 forwards) | 94.7% | 96.0% |
| VAE decode | 10.9 s (5.1%) | 26.8 s (3.9%) |
| Text encoding per prompt | 0.31 s (0.14%) | 0.36 s (0.05%) |
| Mean forward | 2,029 ms | 6,510 ms |
| Kernel split: attention / GEMM / norm + other | 33% / 28% / 39% | 51% / 21% / 28% |
| Of which dtype copies (top-15 kernels only) | 12.5% | 9.0% |
| Achieved: attention / GEMM | 56 / 63 TFLOPS | 60 / 60 TFLOPS |
| Peak allocated: transformer / VAE decode | 4.1 / 12.0 GB | 5.1 / 12.2 GB |
| Mean SM clock (min), power, max temp | 1,181 MHz (945), 71.7 W, 81 °C | 1,195 MHz (885), 71.8 W, 81 °C |

Notes on the table:
- Text encoding is timed separately, before the runs. It is the per-prompt share of encoding the prompt plus the empty negative prompt.
- The kernel split is from one profiled forward of the last prompt.
- Achieved TFLOPS divide the cost model's FLOP count by the measured kernel time.

## Predictions

| # | Prediction | Result | Verdict |
|---|---|---|---|
| B1 | 33 f: 1.6–3.0 min; 81 f: 6–11 min | 3.57 min; 11.31 min | ✗ both slower |
| B2 | Transformer ≥ 90%; text < 1%; VAE 2–8% at 81 f | 94.7–96.0%; ≤ 0.14%; 3.9% | ✓ |
| B3 | 81 f split: attention 55–75%, GEMM 18–32%, norm + other 6–15% | 51%, 21%, 28% | ✗ |
| B4 | Attention share at 33 f ≥ 15 pp lower; GEMM 30–45% | 17.8 pp lower; GEMM 27.6% | ✗ (GEMM just below the range) |
| B5 | Forward ratio 81 / 33 f: 3.0–4.2× | 3.21× | ✓ |
| B6 | Weights 14–15.5 GB; peak < 22 GB without offload | 14.71 GB; out of memory | ✗ |
| B7 | Mean SM clock ≤ 1,800 MHz | 1,175–1,197 MHz | ✓ |

3 of 7 hold.

## What the misses show

**1. Elementwise work is 3.5× what the cost model assumed, because the residual path runs in fp32.**
- Measured norm + other time per forward is 1,844 ms at 81 frames. The model predicted 520 ms. The ratio is the same at 33 frames (784 ms against 220 ms).
- The kernel names explain it: `CUDAFunctor_add<float>`, `BinaryFunctor<float, float, …>` and `layer_norm_kernel<float, float>`. diffusers' `WanTransformerBlock` upcasts to fp32 for LayerNorm, adaLN modulation and the gated residual adds, then casts back to bf16.
- So every elementwise pass moves 4-byte values, and the casts add copy kernels of their own. The copies alone are 9–12.5% of the forward.
- A rough check: the fp32 residual add takes 312 ms per forward. At 231 GB/s, one N×1536 fp32 add (three tensor passes) takes 2.6 ms. That makes about 4 adds per block, which fits the block structure and means these kernels run near bandwidth.

**2. The clock is about 58% of boost, because the card sits at its 72 W cap.**
- Datasheet dense bf16 is 121 TFLOPS at 2,040 MHz. At the measured ~1,190 MHz the scaled peak is about 70.6 TFLOPS.
- Flash attention and the GEMMs both reach about 60 TFLOPS, which is about 85% of that clock-scaled peak.
- They are already efficient. The cost model's "55–60% of datasheet" happened to be about right, but for the wrong reason.

**3. Attention dominates less than predicted.** At 81 frames the attention share falls short of the predicted range because the fp32 elementwise path takes the time the model gave to attention. The length effect holds: the attention share is 33% at 33 frames and 51% at 81 frames.

**4. Memory is limited by the VAE, not the transformer.** The transformer never exceeds 5.1 GB. The fp32 VAE decode needs about 8.8 GB of activations. That, plus the text encoder, is what ran out of memory.

## Consequences for the optimization plan

Measured ceilings, assuming the touched kernels get exactly their ideal factor:

| Change | Touches (81 f / 33 f) | Ceiling 81 f | Ceiling 33 f |
|---|---|---|---|
| Remove all norm + elementwise time (upper bound for fusion / `torch.compile`) | 28% / 39% | 1.39× | 1.64× |
| 8-bit attention at 2× | 51% / 33% | 1.34× | 1.20× |
| W8A8 FP8 GEMMs at 2× | 21% / 28% | 1.12× | 1.16× |

- **The Triton fusion kernel is worth much more than planned.** The cost model put it at about 1%; the measured elementwise share is 28–39%. A good target is one kernel that reads bf16, does LayerNorm in fp32 registers, applies `(1+scale)·x+shift`, and writes bf16. The gated residual add is a second target. The comparison with `torch.compile` matters more now, because Inductor targets exactly this kind of work.
- **8-bit attention is still the largest single lever at 81 frames.** Fusion is the largest at 33 frames, so the two lengths rank the changes differently.
- **FP8 gains may be capped by power, not by tensor-core throughput.** The card is already at 72 W. Predict this before the FP8 run.
- The `nsys` file in this zip comes from the earlier out-of-memory run (its timestamp is before the fix), so it shows no complete generation. Re-run that cell if a timeline is needed.
