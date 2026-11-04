# torch.compile vs eager on Colab L4 (2026-11-04)

Each `WanTransformerBlock` is compiled with diffusers' `compile_repeated_blocks(fullgraph=True)` in Inductor default mode, with no CUDA graphs and no GEMM autotuning. Everything else is the same as the baseline (`L4_baseline.md`). Software is also the same: torch 2.11.0+cu130, diffusers 0.42.1. Eager and compiled runs were made in the same Colab session.

Raw data:
- `eager_33f/`, `compile_33f/`: 4 prompts each.
- `eager_81f/`, `compile_81f/`: 1 prompt each.
- `compile_*/compare.json`: frame comparison against eager.

Predictions: `../predictions_compile_2026-11-03.md`.

## Measurements

| | 33 f eager | 33 f compiled | 81 f eager | 81 f compiled |
|---|---|---|---|---|
| Mean forward | 2,027 ms | 1,511 ms (**1.34×**) | 6,497 ms | 5,286 ms (**1.23×**) |
| Wall time per video | 214.0 s | 162.4 s (1.32×) | 677.3 s | 556.0 s (1.22×) |
| Flash attention kernel | 673 ms | 733 ms (+8.9%) | 3,327 ms | 3,536 ms (+6.3%) |
| GEMM kernels | 557 ms | 598 ms (+7.3%) | 1,350 ms | 1,413 ms (+4.7%) |
| Everything else (norm, elementwise, copies) | 784 ms | 165 ms | 1,844 ms | 421 ms |
| Mean SM clock | 1,183 MHz | 1,100 MHz (−7.0%) | 1,199 MHz | 1,165 MHz (−2.8%) |
| Peak allocated in transformer | 4.11 GB | 3.82 GB | 5.04 GB | 4.35 GB |
| Compile cost (compiled warmup − eager warmup) | | 12.2 s | | 5.2 s |

Notes on the table:
- Kernel times are from one profiled forward.
- In the compiled runs, "everything else" includes one Inductor kernel the classifier filed under attention because its name contains `scaled_dot_product_flash_attention`. It is the fused q/k RMSNorm, 6.6 ms at 33 f and 22.4 ms at 81 f.
- Run to run, the four 33-frame compiled runs differ by at most 0.6 s.

Quality, compiled vs eager, same seed:

| Video | Mean PSNR (worst frame) | LPIPS | Max abs pixel diff |
|---|---|---|---|
| 33 f, fox | 29.5 dB (28.1) | 0.022 | 244 |
| 33 f, coast | 27.2 dB (26.7) | 0.067 | 215 |
| 33 f, leaf | 29.2 dB (28.7) | 0.036 | 236 |
| 33 f, night market | 25.4 dB (24.7) | 0.080 | 254 |
| 81 f, fox | 21.6 dB (19.6) | 0.101 | 255 |

Eager in this session against the baseline run from the day before (`eager_33f/compare_vs_base_33f.json`): all four videos are **bit-identical**.

## Predictions

| # | Prediction | Result | Verdict |
|---|---|---|---|
| C1 | Forward speedup 33 f 1.21–1.41×, 81 f 1.14–1.27× | 1.34×, 1.23× | ✓ |
| C2 | 33 f elementwise 200–430 ms; copies < 3% | 165 ms; aten copies < 0.1% | ✗ (better than predicted) |
| C3 | Attention and GEMM change < ±6% | flash +8.9% / +6.3%, GEMM +7.3% / +4.7% | ✗ |
| C4 | SM clock 0–6% lower than eager | −7.0% at 33 f, −2.8% at 81 f | ✗ at 33 f, ✓ at 81 f |
| C5 | Compile cost 0.5–4 min | 12 s and 5 s | ✗ (much cheaper) |
| C6 | PSNR ≥ 30 dB and LPIPS ≤ 0.05 on every video | 21.6–29.5 dB, LPIPS 0.022–0.101 | ✗ |
| C7 | Eager reproduces the baseline bit for bit across sessions | identical | ✓ |

2 of 7 hold.

## What the results show

**1. Inductor removed almost all of the fp32 elementwise path.** At 33 frames the elementwise time dropped from 784 ms to 165 ms per forward, about 4.8× less. Each upcast → LayerNorm → modulate → downcast chain became one `triton_red_fused__to_copy_add_mul_native_layer_norm` kernel, and the aten dtype copies are gone. The biggest remaining piece is GELU over N×8960, which Inductor does not fuse into the GEMM in default mode. At 33 frames it is 58 ms, close to what one bf16 read and write of that tensor costs at 231 GB/s (65 ms).

**2. Under the 72 W cap, part of the saving goes back to the compute kernels.**
- With the memory-bound phases gone, flash attention and the GEMMs run for a larger part of the time and the clock drops.
- At 33 frames: the clock drops 7.0%, which should make compute kernels about 7.6% slower. Flash measured +8.9% and GEMM +7.3%, so the clock explains it.
- At 81 frames the clock drops only 2.8%, but flash slows 6.3%, so the clock explains about half. The clock is averaged over the whole run (nvidia-smi samples every 500 ms), while the kernel times come from one profiled forward, so the two do not line up exactly.
- In total, 16% (33 f) and 19% (81 f) of the elementwise time saved was lost to slower compute kernels.
- C4 predicted the direction correctly. C3 failed for the same reason.

**3. Compiled and eager outputs diverge far more than rounding error, and one compiled video has a visible defect.**
- Eager is deterministic, even across sessions (C7). So the whole difference comes from fused fp32 math rounding differently, amplified over 50 sampling steps.
- The compiled videos show the same scenes with the same composition, but small details differ: positions in the crowd, lantern shapes, the fox's pose.
- In the last frame of the 81-frame compiled fox video there is a detached extra tail floating above the fox. The eager video has no such defect. See `figures/compile_81f_fox_last_frame.png` (eager left, compiled right) and `figures/compile_vs_eager_frames.png`, where the rows alternate eager and compiled for the night market (33 f), the coast (33 f) and the fox (81 f).
- With one sample per setting, this cannot tell "compile makes quality worse" apart from "compile lands on a different, equally likely sample that happened to have a defect". PSNR and LPIPS against eager measure how far the trajectory moved, not quality.

## Next

- **Rounding noise floor (needed before claiming any optimization keeps quality).** Run eager with a second attention backend that is mathematically equivalent but rounds differently, for example SDPA memory-efficient instead of flash. Compare it against flash eager in the same way. If that also gives about 25–30 dB, then compile is within the noise floor, and quality has to be judged with prompt-level metrics (CLIP score) and more seeds, not with pixel distance to eager.
- **Compiled forward split** at 81 f: flash 66%, GEMM 26%, rest 8%. At 33 f: 49%, 40%, 11%. With compile as the new baseline, the ceilings are:

| Change on top of compile | 81 f | 33 f |
|---|---|---|
| Attention 2× faster (8-bit attention) | 1.49× | 1.32× |
| GEMM 2× faster (W8A8 FP8) | 1.15× | 1.25× |
| Remove all remaining elementwise | 1.09× | 1.12× |

- **The hand-written Triton fusion kernel now has little end-to-end headroom** (≤ 1.09–1.12×). Keep it as a kernel-level comparison against Inductor's fused LayerNorm kernel (GB/s reached), or retarget it at GELU fused into the up-projection.
- **8-bit attention is the largest remaining lever**, especially at 81 frames.
