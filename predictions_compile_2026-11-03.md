# torch.compile predictions before the first run (2026-11-03)

Written before running `colab/02_compile.ipynb`. The setup is the same as the baseline (`results/L4_baseline.md`) except for one change: each `WanTransformerBlock` is compiled with Inductor, using diffusers' `compile_repeated_blocks(fullgraph=True)` in default mode. Default mode means no CUDA graphs and no GEMM autotuning, so the change is kernel fusion only. Eager is re-run in the same Colab session, so the A/B comparison uses the same GPU instance.

## What the change can touch

From the baseline kernel split:
- Norm + elementwise time is 784 ms of a 2,014 ms forward at 33 frames (39%) and 1,844 ms of 6,562 ms at 81 frames (28%).
- About a third of that is dtype copies, from the fp32 upcasts in the block.
- Inductor should fuse each upcast → LayerNorm → `(1+scale)·x+shift` → downcast chain, and each gated residual add, into one kernel per chain.
- It does not replace SDPA (flash attention stays as is) or the cuBLAS GEMMs in default mode.
- CPU overhead was < 1 s per video, so CUDA graphs would have nothing to remove.

If fusion leaves 25–55% of the elementwise time:

| | Ceiling (all elementwise gone) | Predicted |
|---|---|---|
| 33 frames | 1.64× | 1.21–1.41× |
| 81 frames | 1.39× | 1.14–1.27× |

## Predictions

| # | Prediction | Would be wrong if |
|---|---|---|
| C1 | Mean forward speedup over eager (same session): 33 f 1.21–1.41×, 81 f 1.14–1.27× | either outside its range |
| C2 | Norm + elementwise kernel time per forward at 33 f drops from ~784 ms to 200–430 ms; dtype-copy kernels drop below 3% of the forward | outside 200–430 ms, or copies ≥ 3% |
| C3 | Attention and GEMM kernel times change by less than ±6% | either changes by ≥ 6% |
| C4 | Mean SM clock under compile is 0–6% lower than eager. With the low-power memory-bound phases removed, tensor-core kernels fill more of the 72 W budget. | clock higher than eager, or > 6% lower |
| C5 | Compile cost (first short run minus the same run in eager) is 0.5–4 min, paid once per process | outside 0.5–4 min |
| C6 | Output vs eager with the same seed: not bit-identical (fused fp32 math rounds differently, and 50 steps amplify it), but close: mean PSNR ≥ 30 dB and LPIPS ≤ 0.05 on every video | any video below 30 dB or above 0.05 LPIPS |
| C7 | Eager re-run in a new session reproduces the downloaded baseline frames bit for bit (same GPU type, same software) | any pixel differs |

C7 is checked locally against `results/base_33f/frames_*.npy` from the baseline zip, using the eager frames from this run.
