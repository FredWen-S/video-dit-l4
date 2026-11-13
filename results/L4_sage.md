# 8-bit attention (SageAttention2) on top of compile, L4 (2026-11-13)

Predictions: `predictions_sage_2026-11-11.md`. Notebook: `colab/05_sage_attention.ipynb`. This is the third attempt; the first is in `L4_sage_attempt1.md`.

**Result: 7 of 8 predictions held.**
- **Speed:** SageAttention with FP8 P·V makes the forward 1.18× faster at 33 frames and 1.35× faster at 81 frames, on top of compile.
- **Quality:** it meets the pre-set bar of being no further from fp32 than bf16 eager. On the market scene it clears that bar by only 0.4 dB, and it gives up about 3 dB against compile.
- **fp16 P·V variant:** keeps compile's distance to fp32 almost exactly, and is 1.09× faster at 33 frames.
- **The miss (S5):** on the fox, Sage changed the output less than the predicted range.

## Setup

- **Environment:** torch 2.14.1+cu130, diffusers 0.42.1 (pinned), transformers 5.22.2, SageAttention 2.2.0 built from source (commit d1a57a5) for sm_89.
- **Settings:** per-block compile with fullgraph, 50 steps, CFG 5.0, seed 0, 480×832. All runs were in one session.
- **Getting Sage to build and compile:**
  - **C++ standard:** torch 2.14's headers use C++20 `requires` clauses, and SageAttention's `setup.py` builds as C++17. The notebook patches the flag to `-std=c++20`.
  - **fullgraph compile:** `sageattn`'s Python dispatch calls `torch.cuda.device_count()`, which Dynamo cannot trace. `bench/sage_ops.py` wraps the whole Sage call in one `torch.library` custom op (`videodit::sage_attn`) and swaps it into diffusers' attention dispatch. This keeps fullgraph compile, so attention is the only thing that changes. The other option, `fullgraph=False`, would have split every block at attention.
  - **Self-test:** before the runs, the custom op is checked against eager `sageattn`, both directly and under fullgraph compile.
- **Attention calls:** both attention calls in each block go through Sage: self-attention (14,040 or 32,760 tokens) and cross-attention against 512 text tokens. That is 60 calls per forward.

## Verdicts

| # | Prediction | Measured | |
|---|---|---|---|
| S1 | compile and fp32 reproduce the stored hashes when torch is 2.14.1 | compile 4 of 4 identical to `compile_rerun_33f`; fp32 2 of 2 identical to `fp32_33f` | ✓ |
| S2 | microbenchmark: Sage 1.4–2.0× faster than flash at 14,040 and 1.5–2.2× at 32,760; error 3–20× flash's; K bias raises error < 2× | 1.59× and 1.69×; error 16.8× (3.9e-2 vs 2.3e-3); K bias ×1.00 | ✓ |
| S3 | end to end 1.14–1.30× at 33 f and 1.25–1.50× at 81 f | **1.18×** and **1.35×** | ✓ |
| S4 | attention + quantization per forward: 33 f 714 → 375–510 ms, 81 f 3,536 → 1,770–2,360 ms; clock within ±4% | 726 → 485 ms; 3,564 → 2,201 ms; clock −2.4% and −0.3% | ✓ |
| S5 | Sage vs compile at 33 f: PSNR 20–30 dB and LPIPS ≤ 0.12 on every video | 33.4 / 29.6 / 26.0 / 25.2 dB, LPIPS ≤ 0.088. **Fox 33.4 dB is above the range** | ✗ |
| S6 | Sage vs fp32: fox ≥ 27.1 dB, market ≥ 22.8 dB | fox 32.0, market **23.2** | ✓ |
| S7 | fp16-PV closer to compile than FP8-PV on ≥ 3 of 4, and its attention 1.1–1.4× faster than flash | 4 of 4 closer (by 3.2–7.7 dB); attention 1.22× | ✓ |
| S8 | 81 f fox, Sage vs compile: 17–28 dB | 27.8 dB | ✓ |

On S1: under diffusers 0.43.0 in attempt 1, compile did not reproduce. With 0.42.1 pinned, it reproduces bit for bit. The mismatch therefore came from the diffusers upgrade, not from compile being unstable across sessions. It changed compile's bf16 output and left fp32 eager unchanged.

On S5: I gave the same range to every video, but the fox is a simple scene. It also moved the least under compile vs eager (29.5 dB, the highest of the four). The other three videos and every LPIPS value are inside the range.

## Speed

Mean forward over the 100 forwards of each video; kernel times are from one profiled forward.

| | Mean forward | vs compile | SM clock | Wall per video |
|---|---|---|---|---|
| 33 f compile (flash) | 1,494 ms | 1.00× | 1,109 MHz | 160.6 s |
| 33 f Sage, FP8 PV | 1,269 ms | **1.18×** | 1,082 MHz | 138.2 s |
| 33 f Sage, fp16 PV | 1,365 ms | 1.09× | 1,097 MHz | 147.8 s |
| 81 f compile (flash) | 5,280 ms | 1.00× | 1,160 MHz | 555.4 s |
| 81 f Sage, FP8 PV | 3,924 ms | **1.35×** | 1,156 MHz | 419.8 s |

Kernel time per forward, in ms:

| | Attention kernel | Sage preparation | GEMM | Norm (Inductor) | Other |
|---|---|---|---|---|---|
| 33 f compile | 726.4 | – | 598.0 | 79.2 | 75.7 |
| 33 f Sage FP8 PV | 425.0 | 60.4 | 610.0 | 79.8 | 75.7 |
| 33 f Sage fp16 PV | 546.8 | 49.5 | 600.0 | 79.8 | 75.7 |
| 81 f compile | 3,563.6 | – | 1,410.4 | 202.4 | 185.8 |
| 81 f Sage FP8 PV | 2,046.9 | 153.7 | 1,340.9 | 203.3 | 185.8 |

"Sage preparation" is everything Sage adds:
- quantizing Q and K to INT8, per thread;
- the transpose, pad and permute pass on V, and its FP8 scaling (FP8-PV only);
- K smoothing, which is a mean over tokens and a subtraction, run as generic ATen kernels.

It costs 4–5% of a forward at 33 f and 3% at 81 f. The prediction file estimated about 30 ms at 33 f, so this is twice that.

- **The attention kernel alone** is 1.71× faster than flash at 33 f and 1.74× at 81 f, against an ideal 2×. With preparation included it is 1.50× and 1.62×.
- **The fp16-PV kernel** is 1.33× faster than flash, matching its ideal of about 1.33×. Only the QKᵀ product gets the doubled rate there.
- **Where the gain goes.** The forward-time savings match the attention savings: at 33 f, 241 ms saved in attention and 225 ms in the forward. The GEMMs ran 2% slower at 33 f, the same as the 2.4% drop in clock. At 81 f they ran 5% faster at the same mean clock; the clock during the single profiled forward was not recorded.
- **Against the ceiling.** The ceiling if attention were exactly 2× faster was 1.32× at 33 f and 1.50× at 81 f. Sage reached 1.18× and 1.35×. The missing part is the gap between the kernel's 1.7× and 2×, plus the preparation passes.
- **Isolated benchmark vs in the model.** In isolation, Sage including quantization is 1.59× and 1.69× faster than flash. In the model it is 1.50× and 1.62×, so the isolated benchmark overstates the gain slightly, as attempt 1 suggested it might.
- **Memory.** Transformer peak memory is unchanged: 3.82 GB at 33 f and 4.35 GB at 81 f.

At 81 f, VAE decode (26.8 s) is now 6.4% of the wall time per video, up from 4.8%.

## Quality

All metrics are computed from the exact frames (npy); the mp4s are not used for metrics. For reference, the exact attention-kernel swap (flash vs memory-efficient) gives 36.2 dB at 33 f, and compile vs eager gives 25.4–29.5 dB.

Against compile, the session baseline, PSNR in dB and LPIPS:

| Video | Sage FP8 PV | Sage fp16 PV |
|---|---|---|
| 0 fox | 33.4 / 0.013 | 36.6 / 0.006 |
| 1 coast | 29.6 / 0.041 | 37.3 / 0.009 |
| 2 | 26.0 / 0.088 | 29.4 / 0.033 |
| 3 market | 25.2 / 0.076 | 30.9 / 0.034 |
| 81 f fox | 27.8 / 0.037 | not run |

Against fp32 (bf16 weights upcast, TF32 off), the quality reference since `L4_compile_diagnosis.md`:

| | Fox | Market |
|---|---|---|
| Compile (flash) | 35.6 dB / 0.007 | 26.2 dB / 0.077 |
| Sage, fp16 PV | 35.4 dB / 0.007 | **26.6 dB** / 0.062 |
| Sage, FP8 PV | 32.0 dB / 0.015 | 23.2 dB / 0.121 |
| Eager bf16 (earlier session, same frames) | 28.1 dB / 0.029 | 23.8 dB / 0.110 |
| Bar: eager minus 1 dB | 27.1 dB | 22.8 dB |

What this shows:
- **The quality cost is the FP8 P·V product.** With fp16 P·V, INT8 Q·Kᵀ costs nothing measurable. It is 0.2 dB further from fp32 on the fox and 0.5 dB closer on the market.
- **The microbenchmark agrees.** fp16-PV's relative error is 1.2e-2 against FP8-PV's 3.9e-2.
- **FP8 P·V moves the output away from fp32**, by about 3 dB compared with compile. On the fox, FP8-PV is still closer to fp32 than eager is (32.0 vs 28.1 dB). On the market it is slightly further (23.2 vs 23.8 dB) and clears the bar by 0.4 dB.

![market, last frame](figures/sage_market_last_frame.png)

`figures/sage_market_last_frame.png`, from left: fp32, compile, Sage FP8 PV, Sage fp16 PV, market, last frame. The image is taken from the mp4s, so it is for viewing only.
- In the FP8-PV frame, the nearest lantern is larger and closer, with characters on it.
- The street light that fp32 and compile both have is gone.
- The lanterns further down the street have moved.
- The fp16-PV frame keeps the street light and the lantern layout.

`figures/sage_fox_last_frame.png` (same order) shows no visible difference.

There is no fp32 reference at 81 f, so the 81 f quality is judged only against compile: 27.8 dB. For comparison, compile moved the 81 f output 21.6 dB away from eager.

## What this means

By the rule written before the run ("S3 and S6 hold → it becomes part of the baseline"), **compile + SageAttention (FP8 P·V) is the new speed baseline**: 1.35× at 81 frames on top of compile's 1.23×.

Two caveats go with it:
- The market scene clears the bar by only 0.4 dB, and its content changes visibly.
- The fp16-PV variant is close to lossless. It is 1.09× at 33 f; at 81 f it was not run. Scaling its 33 f kernel ratio gives an estimate of about 1.18× at 81 f.

If a later step needs headroom on quality (step caching, fewer steps), fp16-PV is the variant to stack on.

## Files

- `compile_33f_s3/`, `sage_33f/`, `sage_fp16pv_33f/`: 33 f, 4 prompts, with `compare.json` (vs compile) and `compare_vs_fp32.json`.
- `compile_81f_s3/`, `sage_81f/`: 81 f fox.
- `fp32_33f_s3/`: report and hashes only (identical to `fp32_33f/`).
- `sage_micro.json`: the microbenchmark.
- `figures/sage_market_last_frame.png`, `figures/sage_fox_last_frame.png`.
- **Kernel split fix.** The `kernels` field of these five reports was recomputed from the profiler traces with the fixed classifier in `profile_baseline.py`. The original split filed Sage's attention kernel under `attn_quant`, because its name contains `QuantGranularity`. Inductor's RMSNorm kernel, whose name includes the attention op it feeds, now counts as norm rather than attention (6.6 ms at 33 f). K smoothing appears under "other".
