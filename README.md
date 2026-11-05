# video-dit-l4

Inference profiling and optimization of a video diffusion transformer (Wan2.1-T2V-1.3B) on a single 24 GB L4. The method is the same as in `lowbit-decode-a53` and `lowbit-decode-gpu`: break the time down first, write predictions before each run, and change one variable at a time. Every speedup is reported together with a quality comparison against the baseline output.

## Status

- [x] Analytic cost model (`tools/cost_model.py`) and baseline predictions (`predictions_baseline_2026-11-03.md`)
- [x] Baseline script: phase timing, peak memory, clocks, kernel-category split (`bench/profile_baseline.py`)
- [x] Baseline run on Colab L4: 3 of 7 predictions held (`results/L4_baseline.md`). Elementwise work in fp32 takes 28–39% of a forward, so the fusion kernel moves up the plan
- [ ] Optimizations, one at a time
  - [x] `torch.compile` (per-block, default mode): 1.34× at 33 f and 1.23× at 81 f; 2 of 7 predictions held (`results/L4_compile.md`). Outputs diverge from eager (21–29 dB), so the next step is a rounding noise floor
- [x] Rounding noise floor (`results/L4_noise_floor.md`): an exact attention-kernel swap moves the output to 36–39 dB and a different seed to 10–13 dB. Compile sits in between at 22–30 dB, so its divergence is not rounding noise. 2 of 6 measured predictions held. Next: find what compile changes (`emulate_precision_casts`, full-precision reference)
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
