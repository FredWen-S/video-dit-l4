#!/usr/bin/env python3
"""Why does tensorwise FP8 lose 6 dB on the fox when the microbenchmark showed no difference?

Hooks every nn.Linear in the 30 transformer blocks (eager bf16, Sage fp16-PV, no FP8) and, on a
few sampled forwards, measures per linear:
- activation spread: global amax, quantiles of the per-token (row) amax, and max / median of the
  per-channel (column) amax
- the share of nonzero inputs that land below e4m3's smallest normal (2^-6) after scaling, i.e.
  that keep fewer than 3 mantissa bits, with a tensor scale vs a row scale
- the GEMM output error of four FP8 combinations against the bf16 inputs in fp32, emulated as
  dequantize-then-matmul in fp32 (what _scaled_mm computes with fp32 accumulation):
  act tensor / weight tensor (tt), act row / weight row (rr), act row / weight tensor (rt),
  act tensor / weight row (tr). Which of rt and tr moves towards tt says whether the activations'
  or the weights' single scale costs the accuracy.

Prints a per-layer-type summary and the worst (block, layer) pairs, and writes JSON.

usage: python bench/fp8_act_stats.py --prompt-ids 0 3 --out results/g_act_stats.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(__file__))
from fp8_linear import quantize  # noqa: E402

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MIN_NORMAL = 2.0 ** -6


def deq(x, rowwise):
    q, s = quantize(x, rowwise)
    return q.float() * s


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def stats(x, w):
    """x (M, K) bf16 input, w (N, K) bf16 weight."""
    xf, wf = x.float(), w.float()
    ra = xf.abs().amax(dim=1)                       # per token
    ca = xf.abs().amax(dim=0)                       # per channel
    g = ra.max()
    nz = xf != 0
    n_nz = nz.sum().clamp(min=1)
    sub_t = ((xf.abs() / (g / 448.0) < MIN_NORMAL) & nz).sum() / n_nz
    sub_r = ((xf.abs() / (ra[:, None].clamp(min=1e-12) / 448.0) < MIN_NORMAL) & nz).sum() / n_nz
    q = torch.quantile(ra, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], device=ra.device))
    ref = xf @ wf.t()
    xt, xr, wt, wr = deq(x, False), deq(x, True), deq(w, False), deq(w, True)
    wra = wf.abs().amax(dim=1)
    return dict(
        M=x.shape[0], K=x.shape[1], N=w.shape[0],
        act_amax=g.item(), row_amax_q=[v.item() for v in q],
        row_spread=(g / q[2].clamp(min=1e-12)).item(),      # global amax / median token amax
        col_spread=(ca.max() / ca.median().clamp(min=1e-12)).item(),
        w_row_spread=(wra.max() / wra.median().clamp(min=1e-12)).item(),
        act_sub_tensor=sub_t.item(), act_sub_row=sub_r.item(),
        act_err_tensor=rel(xt, xf), act_err_row=rel(xr, xf),
        w_err_tensor=rel(wt, wf), w_err_row=rel(wr, wf),
        out_tt=rel(xt @ wt.t(), ref), out_rr=rel(xr @ wr.t(), ref),
        out_rt=rel(xr @ wt.t(), ref), out_tr=rel(xt @ wr.t(), ref),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-ids", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--forwards", type=int, nargs="+", default=[0, 1, 40, 41, 98, 99],
                    help="transformer calls to measure (two per step with CFG: even = cond, odd = uncond)")
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from diffusers import AutoencoderKLWan, WanPipeline
    import sage_ops

    with open(os.path.join(os.path.dirname(__file__), "prompts.txt")) as f:
        prompts = [p.strip() for p in f if p.strip() and not p.startswith("#")]
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    sage_ops.install()
    pipe.transformer.set_attention_backend("_sage_qk_int8_pv_fp16_cuda")
    # Encode up front and drop the 11 GB text encoder, as profile_baseline.py does, to leave room for
    # the fp32 emulation (an FFN input in fp32 is 0.5 GB)
    with torch.no_grad():
        embeds = {pid: pipe.encode_prompt(prompt=prompts[pid], do_classifier_free_guidance=True,
                                          max_sequence_length=512, device="cuda") for pid in args.prompt_ids}
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    torch.cuda.empty_cache()

    state = dict(call=-1, prompt=None)
    rows = []

    def count(_m, _a, _k):
        state["call"] += 1
    pipe.transformer.register_forward_pre_hook(count, with_kwargs=True)

    for b, block in enumerate(pipe.transformer.blocks):
        for name, mod in block.named_modules():
            if isinstance(mod, torch.nn.Linear):
                def hook(m, inp, b=b, name=name):
                    if state["call"] not in args.forwards:
                        return
                    x = inp[0].reshape(-1, m.in_features)
                    with torch.no_grad():
                        r = stats(x, m.weight)
                    rows.append(dict(prompt_id=state["prompt"], call=state["call"], step=state["call"] // 2,
                                     branch="cond" if state["call"] % 2 == 0 else "uncond",
                                     block=b, layer=name, **r))
                mod.register_forward_pre_hook(hook)

    for pid in args.prompt_ids:
        state.update(call=-1, prompt=pid)
        g = torch.Generator("cuda").manual_seed(0)
        pe, ne = embeds[pid]
        with torch.no_grad():
            pipe(prompt_embeds=pe, negative_prompt_embeds=ne, height=480, width=832, num_frames=args.frames,
                 num_inference_steps=args.steps, guidance_scale=5.0, generator=g, output_type="latent")
        print(f"prompt {pid}: {sum(r['prompt_id'] == pid for r in rows)} linear calls measured")

    # Summary per prompt and layer type, averaged over blocks and sampled forwards
    keys = ["row_spread", "col_spread", "w_row_spread", "act_sub_tensor", "act_sub_row",
            "out_tt", "out_rr", "out_rt", "out_tr"]
    summary = {}
    for pid in args.prompt_ids:
        groups = defaultdict(list)
        for r in rows:
            if r["prompt_id"] == pid:
                groups[r["layer"]].append(r)
        print(f"\n== prompt {pid}: mean over blocks and forwards (spread = max / median)")
        print(f"{'layer':22s} {'tok spread':>10s} {'ch spread':>9s} {'w spread':>8s} {'sub t':>7s} {'sub r':>7s} "
              f"{'out tt':>8s} {'out rr':>8s} {'out rt':>8s} {'out tr':>8s}")
        summary[pid] = {}
        for layer, rs in groups.items():
            m = {k: sum(r[k] for r in rs) / len(rs) for k in keys}
            m["tok_spread_max"] = max(r["row_spread"] for r in rs)
            summary[pid][layer] = m
            print(f"{layer:22s} {m['row_spread']:10.1f} {m['col_spread']:9.1f} {m['w_row_spread']:8.1f} "
                  f"{m['act_sub_tensor']:7.3f} {m['act_sub_row']:7.3f} {m['out_tt']:8.2e} {m['out_rr']:8.2e} "
                  f"{m['out_rt']:8.2e} {m['out_tr']:8.2e}")
        worst = sorted((r for r in rows if r["prompt_id"] == pid), key=lambda r: -r["out_tt"] / r["out_rr"])[:10]
        print("worst tt/rr:")
        for r in worst:
            print(f"  block {r['block']:2d} {r['layer']:20s} step {r['step']:2d} {r['branch']:6s} tt/rr "
                  f"{r['out_tt'] / r['out_rr']:6.2f} | tok spread {r['row_spread']:8.1f} "
                  f"(row amax p1 {r['row_amax_q'][1]:.3g}, p50 {r['row_amax_q'][2]:.3g}, max {r['row_amax_q'][4]:.3g})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(forwards=args.forwards, summary=summary, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
