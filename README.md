# video-dit-l4

Inference profiling and optimization of a video diffusion transformer (Wan2.1-T2V-1.3B) on a single 24 GB L4. The method is the same as in `lowbit-decode-a53` and `lowbit-decode-gpu`: break the time down first, write predictions before each run, and change one variable at a time. Every speedup is reported together with a quality comparison against the baseline output.

## Status

- [x] Analytic cost model (`tools/cost_model.py`) and baseline predictions (`predictions_baseline_2026-11-03.md`)
- [x] Baseline script: phase timing, peak memory, clocks, kernel-category split (`bench/profile_baseline.py`)
- [x] Baseline run on Colab L4: 3 of 7 predictions held (`results/L4_baseline.md`). Elementwise work in fp32 takes 28–39% of a forward, so the fusion kernel moves up the plan
- [ ] Optimizations, one at a time
  - [x] `torch.compile` (per-block, default mode): 1.34× at 33 f and 1.23× at 81 f; 2 of 7 predictions held (`results/L4_compile.md`). Outputs diverge from eager (21–29 dB), so the next step is a rounding noise floor
- [x] Rounding noise floor (`results/L4_noise_floor.md`): an exact attention-kernel swap moves the output to 36–39 dB and a different seed to 10–13 dB. Compile sits in between at 22–30 dB, so its divergence is not rounding noise. 2 of 6 measured predictions held. Next: find what compile changes (`emulate_precision_casts`, full-precision reference)
- [x] What compile changes (`results/L4_compile_diagnosis.md`): all of the divergence is cast elision (emulating eager's casts brings it to the 36 dB floor), and compile is closer to an fp32 reference than eager (35.6 vs 28.1 dB on the fox). 6 of 6 measured predictions held. The quality reference is now fp32; compile (1.37× on torch 2.14) is the new baseline
- [x] 8-bit attention (SageAttention2, INT8 QK + FP8 PV on sm_89) on top of compile, judged against the fp32 reference (`results/L4_sage.md`): 1.18× at 33 f and 1.35× at 81 f; 7 of 8 predictions held. It meets the bar against fp32 (no worse than bf16 eager), with only 0.4 dB to spare on the market scene. The cost is all in FP8 P·V: the fp16-PV variant matches compile's distance to fp32 and is 1.09× at 33 f. Sage runs inside fullgraph compile as a custom op (`bench/sage_ops.py`)
  - Attempt 1 (`results/L4_sage_attempt1.md`): SageAttention did not build, so S2–S8 are untested. Under a diffusers upgrade fp32 eager stayed bit-identical but compile did not. Flash runs about 20% slower inside the model than in isolation. Attempt 2 pins diffusers and stops if the build fails
- [x] Step caching vs fewer steps (`results/L4_step_cache.md`): own first-block cache around the compiled blocks (`bench/step_cache.py`). 2 of 9 predictions held. It skips 43% / 66% / 82% of forwards at 0.05 / 0.10 / 0.20, for 1.6× / 2.4× / 3.6× wall time. 0.05 keeps the 50-step composition but fails the fp32 bar. 0.10 and 20 steps (2.2×) both give a different but clean video, often with the same framing. 0.20 breaks videos. Pixel distance to a 50-step reference can't judge trajectory changes, so the next step is a reference-free quality protocol
- [~] Reference-free quality protocol (`results/L4_quality.md`). 4 prompts × 4 seeds per configuration. 2 of 8 decided predictions held; the blind test is pending. The metrics catch the known break (cache 0.20). CLIP text score ranks 50 steps > cache 0.10 > 20 steps > 10 steps ≈ cache 0.20. Cache 0.10 beats 20 steps by +0.95 CLIP at 2.39× vs 2.24×. Neither is shown to match 50 steps; the loss is mostly on one prompt. The aesthetic score misses the 10-step artifacts, and Laplacian sharpness rises with them, so it is dropped as a quality score
- [ ] Quality metrics: frame hashes (`bench/hash_frames.py`), PSNR/LPIPS (`bench/compare_frames.py`) and CLIP score (`bench/clip_score.py`) are in place

## Plan

**1. Baseline (week 1).** Run 33 frames × 4 prompts and 81 frames × 2 prompts. Record wall time per phase, peak memory, SM clock, and the kernel split of one forward. Capture one `nsys` timeline. Check B1–B7.

**2. One optimization at a time (weeks 1–2).** Write a prediction before each run. The ceilings below come from the cost model at 81 frames and assume each change gets exactly its ideal factor on the part it touches.

| Change | Touches | Ceiling at 81 f | Note |
|---|---|---|---|
| Weight-only int8 / FP8 | nothing compute-bound | ~1.00× | Saves ~1.3 GB. Expected to be a negative result, and worth stating why. |
| W8A8 FP8 linears (torchao, dynamic activation scales) | linear share (24%) | 1.14× | L4 is sm_89 and has FP8 tensor cores. |
| 8-bit attention (SageAttention2 has an sm_89 path) | attention share (65%) | 1.48× | The largest single lever at 81 frames. |
| `torch.compile` | elementwise share + launch overhead | ≤ 1.12× | Compare with the hand-written fusion below. |
| Step caching (own first-block cache vs diffusers' built-in hook) | whole forwards | 1.5–2× if ~half the steps are skipped | Quality cost; sweep the threshold. |
| Fewer sampling steps (50 → 30 / 20) | whole forwards | 1.7× / 2.5× | Quality cost. |

**3. A Triton fusion kernel (week 2).** Wan's blocks use LayerNorm without affine followed by `x·(1+scale)+shift` (adaLN), and RMSNorm on q and k followed by RoPE. Fuse one of these chains into a single kernel. The cost model puts removing two of the 14 activation passes per block at about 1% end-to-end, so the claim to test is at kernel level: bandwidth reached, and whether it beats the fusion `torch.compile` generates for the same chain. Report the end-to-end number too, even if it is about 1%.

**4. Quality (week 3).** Use the same seed and prompts as the baseline. Report per-frame LPIPS and PSNR against the baseline frames, and CLIP score against the prompt. Show a few side-by-side clips. Report negative results as found.

## Running

Colab: open `colab/01_baseline.ipynb` on an L4 runtime.

Locally (no GPU needed):

```
python tools/cost_model.py --frames 81 33 17
```

Compute budget, at about 2 units/hour on an L4: the baseline notebook is roughly 1–1.5 units. Each optimization measured at 33 frames × 4 prompts is about 0.3 units.
