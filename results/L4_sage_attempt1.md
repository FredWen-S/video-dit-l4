# 8-bit attention, attempt 1 on L4 (2026-11-12): SageAttention did not build

Notebook: `colab/05_sage_attention.ipynb` (commit d059d12). Predictions: `predictions_sage_2026-11-11.md`.

## What happened

- SageAttention was not installed. Every report has `sageattention: null`, the microbenchmark has only flash rows, and `sage_33f/`, `sage_81f/` and `sage_fp16pv_33f/` are empty. The zip has no build log, so the cause is not known. The likely one is an nvcc whose major version differs from torch's CUDA (torch 2.14.1+cu130); torch's extension builder refuses that.
- The notebook did not stop after the failed build. The flash runs (compile at 33 f and 81 f, fp32 reference) completed, about 1.5 hours of L4 time.
- The install cell ran `pip install -U diffusers`, which moved diffusers from 0.42.1 to **0.43.0**. torch (2.14.1+cu130) and transformers (5.22.2) are unchanged.

S2–S8 are untested. Attempt 2 changes three things in the notebook:
- diffusers is pinned to 0.42.1.
- The build cell installs a matching nvcc from NVIDIA's apt repository when the versions differ.
- The build cell writes the full log and raises an error when `sageattention` does not import.

## What the flash runs still show

### S1: reproducibility

| Run | Compared with | Result |
|---|---|---|
| fp32 eager, fox and market | `fp32_33f` (diffusers 0.42.1) | identical on 2 of 2 |
| compile, 4 videos | `compile_rerun_33f` (diffusers 0.42.1) | **different on 4 of 4** |

fp32 eager is bit-identical under diffusers 0.43.0, so the scheduler, VAE, text encoder and the transformer's fp32 math did not change. Compile's output did change. There are two candidate causes, and this run cannot tell them apart:

1. **diffusers 0.43.0 changed something in the bf16 path.** The generated Triton kernels have the same names and the same times, so the graph looks the same.
2. **Compile is not bit-stable across sessions.** Inductor can benchmark several configs for a reduction kernel at runtime and keep the fastest. A different winner changes the summation order. Cross-session reproducibility of compile was never tested; the earlier comparison also changed torch.

Attempt 2 pins diffusers 0.42.1 and checks the same hashes. A match points to cause 1; a mismatch points to cause 2.

Since npy files were saved only for fp32, the two compile runs can be compared only by their distance to fp32:

| PSNR to fp32 | Fox | Market |
|---|---|---|
| compile, diffusers 0.42.1 (`compile_rerun_33f`) | 35.57 dB | 26.15 dB |
| compile, diffusers 0.43.0 (`compile_33f_d043`) | 35.09 dB | 26.13 dB |

The difference is at most 0.5 dB. The S5 and S6 comparisons are made within one session against that session's compile run, so they do not rely on bit-exact compile reproduction.

### Compile speed reproduces within the clock

| | Mean forward | SM clock | Attention per forward | GEMM per forward |
|---|---|---|---|---|
| 33 f, earlier session (`compile_rerun_33f`) | 1,474 ms | 1,123 MHz | 720 ms | 591 ms |
| 33 f, this session (`compile_33f_d043`) | 1,503 ms (+1.9%) | 1,100 MHz (−2.1%) | 734 ms | 600 ms |
| 81 f, torch 2.11 (`compile_81f`) | 5,286 ms | 1,165 MHz | 3,559 ms | 1,413 ms |
| 81 f, torch 2.14 (`compile_81f_t214`) | 5,321 ms (+0.7%) | 1,150 MHz (−1.3%) | 3,608 ms | 1,406 ms |

The forward-time differences are no larger than the clock differences. The GPU reached 81 °C in this session against 78 °C in the earlier one. A ±2% run-to-run band at the same settings is the resolution for S3.

The 81 f flash run on torch 2.14 is the S4 starting point for attempt 2: attention is 3,608 ms per forward, 67% of the kernel time.

### Flash in isolation versus inside the model

The microbenchmark (`attn_micro_flash_only.json`, B=1, H=12, D=128) gives:

| Tokens | Flash per call, isolated | Throughput | Flash per block inside the compiled forward |
|---|---|---|---|
| 14,040 (33 f) | 19.1 ms | 63.4 TFLOPS | 24.2 ms |
| 32,760 (81 f) | 104.9 ms | 62.9 TFLOPS | 119.5 ms |

The in-model figure is the flash kernel time per forward divided by 30 blocks. It also includes the small cross-attention call, which is about 4% of the work at 33 f. Even allowing for that, flash runs about 20% slower at 33 f and 14% slower at 81 f inside the model than in isolation. The likely reason is the power cap. The microbenchmark recorded no clocks, so this is a hypothesis.

If the cause is the clock, a kernel speedup measured in isolation (S2) will overstate the end-to-end gain (S3). This is one more reason to measure S4 inside the model.

Flash reaches 63 TFLOPS, about half of the L4's dense bf16 peak (121 TFLOPS at boost clock).

## Files

- `compile_33f_d043/`: compile at 33 f under diffusers 0.43.0 (the notebook named it `compile_33f_s3`), including the hashes and the distance to fp32.
- `compile_81f_t214/`: compile at 81 f on torch 2.14, fox.
- `attn_micro_flash_only.json`: the flash rows of the microbenchmark.
- The fp32 run's hashes match `fp32_33f/`, so it is not stored again. Its npy files (80 MB) are not in the repo; attempt 2 reruns it.
