#!/usr/bin/env python3
"""Per-forward error against fp32: every variant runs one transformer forward on the same inputs.

The end-to-end distance between two videos mixes the size of a numeric change with the luck of
the 50-step trajectory (results/L4_fp8_fast_accum.md). This script removes the trajectory:

  capture  runs the quality-clean base (per-block fullgraph compile + Sage fp16-PV) through the
           pipeline and keeps the transformer's inputs and output at a few sampled steps, both
           CFG branches (even calls = cond, odd = uncond)
  run      loads the transformer as one variant, runs a forward on every captured input, repeats
           the first few, and saves the outputs
  compare  measures each variant's output against fp32's: rel L2 of the velocity (branch), the
           same after 4x4 spatial average pooling (low-pass), and the CFG-combined u + s(c - u)
           (guided); writes per-input rows, per-variant summaries and ranking checks

usage:
  python bench/forward_error.py capture --frames 33 --steps 0 10 20 30 40 49 --out /content/fe33/inputs.pt
  python bench/forward_error.py run --inputs /content/fe33/inputs.pt --variant fp32 --out /content/fe33
  python bench/forward_error.py compare --dir /content/fe33 --report results/h_forward_error_33f.json
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SAGE16 = "_sage_qk_int8_pv_fp16_cuda"

VARIANTS = {
    "fp32": dict(dtype="fp32", attn="efficient"),
    "eager": dict(),
    "compile": dict(compile=True),
    "compile_emul": dict(compile=True, emulate=True),
    "base": dict(compile=True, backend=SAGE16),
    "sage_fp8pv": dict(compile=True, backend="sage"),
    "fp8_row": dict(compile=True, backend=SAGE16, fp8="rowwise"),
    "fp8_row_fast": dict(compile=True, backend=SAGE16, fp8="rowwise", fast=True),
    "fp8_tensor": dict(compile=True, backend=SAGE16, fp8="tensorwise"),
}


def load_prompts():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")) as f:
        return [p.strip() for p in f if p.strip() and not p.startswith("#")]


def env():
    import diffusers
    return dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, diffusers=diffusers.__version__)


# ---------------------------------------------------------------- capture

def cmd_capture(args):
    from diffusers import AutoencoderKLWan, WanPipeline
    import sage_ops

    prompts = load_prompts()
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        embeds = {pid: pipe.encode_prompt(prompt=prompts[pid], do_classifier_free_guidance=True,
                                          max_sequence_length=512, device="cuda") for pid in args.prompt_ids}
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    torch.cuda.empty_cache()

    # The quality-clean base, set up in the same order as profile_baseline.py
    sage_ops.install()
    pipe.transformer.set_attention_backend(SAGE16)
    pipe.transformer.compile_repeated_blocks(fullgraph=True)

    state = dict(call=-1, pid=None)
    records = []

    def hook(_mod, a, kw, out):
        state["call"] += 1
        c = state["call"]
        if c // 2 not in args.steps:
            return
        assert not a, "the pipeline is expected to pass every transformer input by keyword"
        y = out[0] if isinstance(out, (tuple, list)) else out.sample
        records.append(dict(prompt_id=state["pid"], call=c, step=c // 2, branch="cond" if c % 2 == 0 else "uncond",
                            kwargs={k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in kw.items()},
                            out=y.detach().cpu()))
    pipe.transformer.register_forward_hook(hook, with_kwargs=True)

    for pid in args.prompt_ids:
        state.update(call=-1, pid=pid)
        g = torch.Generator("cuda").manual_seed(args.seed)
        pe, ne = embeds[pid]
        t0 = time.perf_counter()
        with torch.no_grad():
            pipe(prompt_embeds=pe, negative_prompt_embeds=ne, height=480, width=832, num_frames=args.frames,
                 num_inference_steps=args.n_steps, guidance_scale=args.guidance, generator=g, output_type="latent")
        assert state["call"] + 1 == 2 * args.n_steps, f"expected {2 * args.n_steps} transformer calls, got {state['call'] + 1}"
        print(f"prompt {pid}: {sum(r['prompt_id'] == pid for r in records)} inputs captured, {time.perf_counter() - t0:.0f} s")

    r0 = records[0]["kwargs"]
    print("captured kwargs:", {k: (tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else v for k, v in r0.items()})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = dict(frames=args.frames, n_steps=args.n_steps, sample_steps=args.steps, prompt_ids=args.prompt_ids,
                guidance=args.guidance, seed=args.seed, env=env())
    torch.save(dict(meta=meta, records=records), args.out)
    print(f"saved {len(records)} inputs to {args.out}")


# ---------------------------------------------------------------- run

def build(variant):
    from diffusers import WanTransformer3DModel

    cfg = VARIANTS[variant]
    tr = WanTransformer3DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")
    if cfg.get("dtype") == "fp32":
        # Same (bf16-rounded) weights as the bf16 variants; only activation and math precision change
        tr.to(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if cfg.get("backend"):
        import sage_ops
        sage_ops.install()
        tr.set_attention_backend(cfg["backend"])
    if cfg.get("fp8"):
        import fp8_linear
        n = fp8_linear.install(tr.blocks, rowwise=cfg["fp8"] == "rowwise", fast_accum=cfg.get("fast", False))
        torch.cuda.empty_cache()
        print(f"fp8 linears ({cfg['fp8']}{', fast accum' if cfg.get('fast') else ''}): replaced {n}")
    if cfg.get("emulate"):
        import torch._inductor.config as inductor_config
        inductor_config.emulate_precision_casts = True
    if cfg.get("compile"):
        tr.compile_repeated_blocks(fullgraph=True)
    ctx = nullcontext
    if cfg.get("attn") == "efficient":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        def ctx():
            return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    return tr, ctx, (torch.float32 if cfg.get("dtype") == "fp32" else torch.bfloat16)


def to_device(kw, dtype):
    out = {}
    for k, v in kw.items():
        if torch.is_tensor(v):
            v = v.to("cuda")
            if v.dtype == torch.bfloat16 and dtype != torch.bfloat16:
                v = v.to(dtype)   # bf16 -> fp32 is exact, so fp32 sees the same input values
        out[k] = v
    return out


def cmd_run(args):
    data = torch.load(args.inputs, weights_only=True)
    records = data["records"]
    tr, ctx, dtype = build(args.variant)

    def fwd(r):
        o = tr(**to_device(r["kwargs"], dtype))
        return (o[0] if isinstance(o, (tuple, list)) else o.sample).detach()

    outs, ms = [], []
    t0 = time.perf_counter()
    with torch.no_grad(), ctx():
        for i, r in enumerate(records):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            y = fwd(r)
            e.record()
            torch.cuda.synchronize()
            ms.append(s.elapsed_time(e))
            outs.append(y.cpu())
            if i == 0:
                print(f"first forward (with compile): {time.perf_counter() - t0:.0f} s")
        repeat_identical = [bool(torch.equal(fwd(records[i]).cpu(), outs[i])) for i in range(min(args.repeat, len(records)))]
    equals_capture = [bool(torch.equal(o, r["out"])) for o, r in zip(outs, records)]
    timed = ms[2:] if len(ms) > 2 else ms
    meta = dict(variant=args.variant, config=VARIANTS[args.variant], n=len(outs), mean_ms=sum(timed) / len(timed),
                repeat_identical=repeat_identical, equals_capture=sum(equals_capture),
                peak_gb=torch.cuda.max_memory_allocated() / 1e9, env=env())
    os.makedirs(args.out, exist_ok=True)
    torch.save(dict(outs=outs), os.path.join(args.out, f"{args.variant}.pt"))
    with open(os.path.join(args.out, f"{args.variant}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{args.variant}: {len(outs)} forwards, {meta['mean_ms']:.0f} ms each | repeats identical "
          f"{sum(repeat_identical)}/{len(repeat_identical)} | equal to the captured base output {sum(equals_capture)}/{len(outs)}")


# ---------------------------------------------------------------- compare

def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def lowpass(y):
    """(B, C, F, H, W) -> 4x4 spatial average pooling."""
    return F.avg_pool3d(y, (1, 4, 4))


def compare(meta_rows, ref, outs_by_variant, guidance):
    """meta_rows: list of dict(prompt_id, step, branch); ref: fp32 outputs; outs_by_variant: {name: outputs}.
    Returns (rows, summary, checks)."""
    pair = defaultdict(dict)                    # (prompt, step) -> branch -> index
    for i, m in enumerate(meta_rows):
        pair[(m["prompt_id"], m["step"])][m["branch"]] = i
    ref = [r.double() for r in ref]
    ref_low = [lowpass(r) for r in ref]
    ref_g = {k: ref[d["uncond"]] + guidance * (ref[d["cond"]] - ref[d["uncond"]])
             for k, d in pair.items() if len(d) == 2}

    rows, summary = [], {}
    for v, outs in outs_by_variant.items():
        outs = [o.double() for o in outs]
        vr = []
        for i, (m, o) in enumerate(zip(meta_rows, outs)):
            vr.append(dict(variant=v, prompt_id=m["prompt_id"], step=m["step"], branch=m["branch"],
                           branch_err=rel(o, ref[i]), low_err=rel(lowpass(o), ref_low[i])))
        guided = {}
        for k, d in pair.items():
            if len(d) == 2:
                g = outs[d["uncond"]] + guidance * (outs[d["cond"]] - outs[d["uncond"]])
                guided[k] = rel(g, ref_g[k])
        rows += vr

        def mean(xs):
            xs = list(xs)
            return sum(xs) / len(xs)
        per_prompt = {p: mean(r["branch_err"] for r in vr if r["prompt_id"] == p) for p in sorted({r["prompt_id"] for r in vr})}
        per_step = {s: mean(r["branch_err"] for r in vr if r["step"] == s) for s in sorted({r["step"] for r in vr})}
        b = mean(r["branch_err"] for r in vr)
        summary[v] = dict(
            branch=b, low=mean(r["low_err"] for r in vr),
            cond=mean(r["branch_err"] for r in vr if r["branch"] == "cond"),
            uncond=mean(r["branch_err"] for r in vr if r["branch"] == "uncond"),
            guided=mean(guided.values()) if guided else None,
            guided_over_branch=(mean(guided.values()) / b) if guided else None,
            low_over_branch=mean(r["low_err"] for r in vr) / b,
            per_prompt=per_prompt, per_step=per_step,
            prompt_spread=max(per_prompt.values()) / min(per_prompt.values()),
            step_spread=max(per_step.values()) / min(per_step.values()),
            guided_per_prompt={p: mean(g for (pp, _), g in guided.items() if pp == p) for p in per_prompt} if guided else None,
        )
    for ref_name in ("eager", "base"):
        if ref_name in summary:
            for v in summary:
                summary[v][f"over_{ref_name}"] = summary[v]["branch"] / summary[ref_name]["branch"]

    # Ranking check: every pair whose overall means differ by > 1.3x keeps its order on every prompt and step
    names = list(summary)
    flips = []
    for i, a in enumerate(names):
        for bname in names[i + 1:]:
            sa, sb = summary[a], summary[bname]
            ratio = max(sa["branch"], sb["branch"]) / min(sa["branch"], sb["branch"])
            if ratio <= 1.3:
                continue
            sign = sa["branch"] > sb["branch"]
            for kind in ("per_prompt", "per_step"):
                for k in sa[kind]:
                    if (sa[kind][k] > sb[kind][k]) != sign:
                        flips.append(dict(a=a, b=bname, overall_ratio=ratio, where=kind, key=k,
                                          a_err=sa[kind][k], b_err=sb[kind][k]))
    checks = dict(order_flips=flips, max_prompt_spread=max(s["prompt_spread"] for s in summary.values()))
    return rows, summary, checks


def cmd_compare(args):
    import math
    inputs = torch.load(args.inputs or os.path.join(args.dir, "inputs.pt"), weights_only=True)
    meta_rows = [dict(prompt_id=r["prompt_id"], step=r["step"], branch=r["branch"]) for r in inputs["records"]]
    ref = torch.load(os.path.join(args.dir, "fp32.pt"), weights_only=True)["outs"]
    names = args.variants or [v for v in VARIANTS if v != "fp32" and os.path.exists(os.path.join(args.dir, f"{v}.pt"))]
    outs = {v: torch.load(os.path.join(args.dir, f"{v}.pt"), weights_only=True)["outs"] for v in names}
    runs = {}
    for v in ["fp32"] + names:
        p = os.path.join(args.dir, f"{v}.json")
        if os.path.exists(p):
            with open(p) as f:
                runs[v] = json.load(f)
    rows, summary, checks = compare(meta_rows, ref, outs, inputs["meta"]["guidance"])

    def db(x):
        return -20 * math.log10(x) if x and x > 0 else float("inf")
    print(f"{'variant':14s} {'branch':>9s} {'dB':>6s} {'/eager':>7s} {'/base':>6s} {'low':>9s} {'guided':>9s} "
          f"{'g/b':>5s} {'prompts max/min':>15s} {'steps max/min':>13s} {'ms':>6s} {'repeat':>7s}")
    for v, s in summary.items():
        r = runs.get(v, {})
        rep = r.get("repeat_identical", [])
        print(f"{v:14s} {s['branch']:9.3e} {db(s['branch']):6.1f} {s.get('over_eager', float('nan')):7.2f} "
              f"{s.get('over_base', float('nan')):6.2f} {s['low']:9.3e} {s['guided'] or float('nan'):9.3e} "
              f"{s['guided_over_branch'] or float('nan'):5.2f} {s['prompt_spread']:15.2f} {s['step_spread']:13.2f} "
              f"{r.get('mean_ms', float('nan')):6.0f} {sum(rep)}/{len(rep)}")
    print("per prompt (branch error):")
    for v, s in summary.items():
        print(f"  {v:14s} " + "  ".join(f"{p}: {e:.3e}" for p, e in s["per_prompt"].items()))
    print("per step (branch error):")
    for v, s in summary.items():
        print(f"  {v:14s} " + "  ".join(f"{k}: {e:.3e}" for k, e in s["per_step"].items()))
    if "base" in runs:
        print(f"base reproduces the captured pipeline outputs: {runs['base']['equals_capture']}/{runs['base']['n']}")
    print(f"order flips among pairs > 1.3x apart: {len(checks['order_flips'])}")
    for fl in checks["order_flips"][:20]:
        print(f"  {fl['a']} vs {fl['b']} ({fl['overall_ratio']:.2f}x overall) flips at {fl['where']} {fl['key']}: "
              f"{fl['a_err']:.3e} vs {fl['b_err']:.3e}")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(dict(meta=inputs["meta"], runs=runs, summary=summary, checks=checks, rows=rows), f, indent=1)
    print(f"wrote {args.report}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--frames", type=int, default=33)
    c.add_argument("--n-steps", type=int, default=50, help="sampling steps of the captured trajectory")
    c.add_argument("--steps", type=int, nargs="+", default=[0, 10, 20, 30, 40, 49], help="steps to keep")
    c.add_argument("--prompt-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    c.add_argument("--guidance", type=float, default=5.0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--inputs", required=True)
    r.add_argument("--variant", choices=list(VARIANTS), required=True)
    r.add_argument("--repeat", type=int, default=4, help="repeat the first n forwards to check run-to-run identity")
    r.add_argument("--out", required=True, help="directory for <variant>.pt and <variant>.json")
    m = sub.add_parser("compare")
    m.add_argument("--dir", required=True)
    m.add_argument("--inputs", default=None, help="default: <dir>/inputs.pt")
    m.add_argument("--variants", nargs="+", default=None)
    m.add_argument("--report", required=True)
    args = ap.parse_args()
    dict(capture=cmd_capture, run=cmd_run, compare=cmd_compare)[args.cmd](args)


if __name__ == "__main__":
    main()
