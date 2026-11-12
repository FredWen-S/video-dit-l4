#!/usr/bin/env python3
"""Baseline text-to-video run of Wan2.1-T2V-1.3B (diffusers) on one GPU.

Per prompt, reports wall time and the split into text encoding, transformer forwards and VAE
decode (CUDA events), peak allocated memory per phase, and SM clock / power sampled from
nvidia-smi (the L4 is power-capped at 72 W, so clocks are part of the result). Each phase is
also an NVTX range, so the same run under `nsys profile` shows them on the timeline.

With --kernels, one transformer forward (captured from the last run) is profiled with
torch.profiler and its GPU time is split into attention / GEMM / norm / other kernels.

All prompts are encoded first, then the umT5 text encoder (~11 GB) is dropped from the GPU:
with it resident, the fp32 VAE decode runs out of memory on the 22 GiB L4 even at 17 frames.

Frames are saved as uint8 .npy so later optimizations can be scored against this baseline.

usage: python bench/profile_baseline.py --frames 33 --steps 50 --kernels --out results/base_33f
       python bench/profile_baseline.py --frames 33 --compile blocks --kernels --out results/compile_33f
       python bench/profile_baseline.py --frames 33 --attn efficient --kernels --out results/efficient_33f
       python bench/profile_baseline.py --frames 33 --compile blocks --attn-backend sage --kernels --out results/sage_33f
       python bench/profile_baseline.py --frames 33 --dtype fp32 --attn efficient --prompt-ids 0 3 --out results/fp32_33f
"""
import argparse
import json
import os
import subprocess
import time
from collections import defaultdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _version(pkg):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


class Phases:
    """Wraps module methods with CUDA events, NVTX ranges and peak-memory tracking."""

    def __init__(self):
        self.events = defaultdict(list)
        self.peak = defaultdict(int)
        self.orig = {}
        self.capture = None   # name of a phase whose next call's (args, kwargs) to keep
        self.captured = None

    def wrap(self, obj, attr, name):
        fn = getattr(obj, attr)
        self.orig[name] = fn

        def wrapped(*a, **k):
            if self.capture == name:
                self.captured = (a, k)
            torch.cuda.reset_peak_memory_stats()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(name)
            s.record()
            out = fn(*a, **k)
            e.record()
            torch.cuda.nvtx.range_pop()
            self.events[name].append((s, e))
            self.peak[name] = max(self.peak[name], torch.cuda.max_memory_allocated())
            return out

        setattr(obj, attr, wrapped)

    def reset(self):
        self.events.clear()
        self.peak.clear()

    def summary(self):
        torch.cuda.synchronize()
        out = {}
        for name, evs in self.events.items():
            ms = [s.elapsed_time(e) for s, e in evs]
            out[name] = dict(calls=len(ms), total_s=sum(ms) / 1e3, mean_ms=float(np.mean(ms)),
                             peak_gb=self.peak[name] / 1e9)
        return out


class SmiSampler:
    """Samples SM clock and power with nvidia-smi in the background."""

    def __enter__(self):
        self.p = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits", "-lms", "500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        rows = []
        for line in self.p.communicate()[0].splitlines():
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                pass
        a = np.array(rows) if rows else np.zeros((0, 3))
        self.stats = {} if not len(a) else dict(
            sm_mhz_mean=float(a[:, 0].mean()), sm_mhz_min=float(a[:, 0].min()),
            power_w_mean=float(a[:, 1].mean()), temp_c_max=float(a[:, 2].max()), samples=len(a))


def kernel_category(name):
    n = name.lower()
    # SageAttention's Q/K/V quantization and padding passes (memory-bound, part of its cost)
    if "quant" in n or "transpose_pad_permute" in n:
        return "attn_quant"
    if any(k in n for k in ("flash", "fmha", "attention", "attn", "sdpa")):
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "cublas", "xmma", "matmul", "_mma")):
        return "gemm"
    if "norm" in n:
        return "norm"
    return "other"


def profile_forward(fn, args, kwargs, path):
    """Profiles one forward and returns GPU time per kernel category, from the chrome trace."""
    with torch.no_grad():
        fn(*args, **kwargs)   # warm
        torch.cuda.synchronize()
        acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=acts) as prof:
            fn(*args, **kwargs)
            torch.cuda.synchronize()
    prof.export_chrome_trace(path)
    with open(path) as f:
        trace = json.load(f)
    by_cat, by_kernel = defaultdict(float), defaultdict(float)
    for ev in trace["traceEvents"]:
        if ev.get("cat") == "kernel":
            by_cat[kernel_category(ev["name"])] += ev["dur"] / 1e3     # us -> ms
            by_kernel[ev["name"]] += ev["dur"] / 1e3
    total = sum(by_cat.values())
    top = sorted(by_kernel.items(), key=lambda kv: -kv[1])[:15]
    return dict(gpu_ms=total, share={k: v / total for k, v in by_cat.items()},
                ms=dict(by_cat), top_kernels=[(k[:120], v, kernel_category(k)) for k, v in top])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts.txt"))
    ap.add_argument("--n-prompts", type=int, default=None, help="use only the first n prompts")
    ap.add_argument("--prompt-ids", type=int, nargs="+", default=None,
                    help="use these prompts (0-based); outputs keep the prompt's index, e.g. frames_3.npy")
    ap.add_argument("--warmup-steps", type=int, default=2, help="untimed short run first; 0 to skip")
    ap.add_argument("--compile", choices=["none", "blocks", "full"], default="none",
                    help="blocks: compile each transformer block (diffusers compile_repeated_blocks); "
                         "full: torch.compile the whole transformer forward")
    ap.add_argument("--no-fullgraph", action="store_true", help="allow graph breaks when compiling")
    ap.add_argument("--emulate-precision-casts", action="store_true",
                    help="Inductor keeps eager's bf16 downcast/upcast pairs inside fused kernels")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="transformer weights and activations; fp32 upcasts the bf16 weights and turns TF32 off, "
                         "and needs --attn efficient (flash has no fp32 kernel)")
    ap.add_argument("--attn-backend", default=None,
                    help="diffusers attention backend for the transformer, e.g. sage or _sage_qk_int8_pv_fp16_cuda "
                         "(set before compiling)")
    ap.add_argument("--attn", choices=["default", "flash", "efficient", "cudnn", "math"], default="default",
                    help="force one SDPA backend inside the transformer only (the VAE keeps the default); "
                         "default lets torch choose, which is flash. math materializes N^2 scores: OOM")
    ap.add_argument("--kernels", action="store_true", help="kernel-category split of one forward")
    ap.add_argument("--no-save-frames", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import diffusers
    import transformers
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    os.makedirs(args.out, exist_ok=True)
    with open(args.prompts) as f:
        all_prompts = [p.strip() for p in f if p.strip() and not p.startswith("#")]
    ids = args.prompt_ids if args.prompt_ids is not None else list(range(len(all_prompts)))[:args.n_prompts]
    prompts = [all_prompts[i] for i in ids]

    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    weights_all_gb = torch.cuda.memory_allocated() / 1e9

    if args.dtype == "fp32":
        # Same (bf16-rounded) weights as the bf16 runs, so only activation and math precision change
        pipe.transformer.to(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if args.attn_backend:
        pipe.transformer.set_attention_backend(args.attn_backend)

    # Compile before wrapping, so the timing wrapper stays outside the compiled region
    if args.emulate_precision_casts:
        import torch._inductor.config as inductor_config
        assert hasattr(inductor_config, "emulate_precision_casts"), "this torch has no emulate_precision_casts"
        inductor_config.emulate_precision_casts = True
    fullgraph = not args.no_fullgraph
    if args.compile == "blocks":
        pipe.transformer.compile_repeated_blocks(fullgraph=fullgraph)
    elif args.compile == "full":
        pipe.transformer.forward = torch.compile(pipe.transformer.forward, fullgraph=fullgraph)

    # Inside the timing wrapper, so the profiled forward (ph.orig) uses the same backend
    if args.attn != "default":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backend = dict(flash=SDPBackend.FLASH_ATTENTION, efficient=SDPBackend.EFFICIENT_ATTENTION,
                       cudnn=SDPBackend.CUDNN_ATTENTION, math=SDPBackend.MATH)[args.attn]
        fwd = pipe.transformer.forward

        def fwd_attn(*a, **k):
            with sdpa_kernel(backend):
                return fwd(*a, **k)
        pipe.transformer.forward = fwd_attn

    ph = Phases()
    ph.wrap(pipe.text_encoder, "forward", "text_encoder")
    ph.wrap(pipe.transformer, "forward", "transformer")
    ph.wrap(pipe.vae, "decode", "vae_decode")

    # Encode every prompt (and the empty negative prompt) up front, then drop the text encoder
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        # max_sequence_length=512 matches WanPipeline.__call__ (encode_prompt alone defaults to 226)
        embeds = [pipe.encode_prompt(prompt=p, do_classifier_free_guidance=args.guidance > 1,
                                     max_sequence_length=512, device="cuda") for p in prompts]
    torch.cuda.synchronize()
    encode = dict(wall_s=time.perf_counter() - t0, **ph.summary()["text_encoder"])
    ph.orig.pop("text_encoder")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    weights_gb = torch.cuda.memory_allocated() / 1e9
    print(f"weights {weights_all_gb:.1f} GB with text encoder, {weights_gb:.1f} GB without | "
          f"encoded {len(prompts)} prompts in {encode['wall_s']:.1f} s")

    def run(i, steps):
        g = torch.Generator("cuda").manual_seed(args.seed)
        pe, ne = embeds[i]
        return pipe(prompt_embeds=pe, negative_prompt_embeds=ne, height=args.height, width=args.width,
                    num_frames=args.frames, num_inference_steps=steps, guidance_scale=args.guidance,
                    generator=g, output_type="np").frames[0]

    # With --compile, the warmup run also pays the compile time; compare it with an eager warmup
    warmup_s = None
    if args.warmup_steps:
        t0 = time.perf_counter()
        run(0, args.warmup_steps)
        torch.cuda.synchronize()
        warmup_s = time.perf_counter() - t0
        print(f"warmup ({args.warmup_steps} steps, compile={args.compile}): {warmup_s:.1f} s")

    results = []
    for i, (pid, prompt) in enumerate(zip(ids, prompts)):
        ph.reset()
        ph.capture = "transformer" if (args.kernels and i == len(prompts) - 1) else None
        torch.cuda.synchronize()
        with SmiSampler() as smi:
            t0 = time.perf_counter()
            frames = run(i, args.steps)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        phases = ph.summary()
        other = wall - sum(p["total_s"] for p in phases.values())
        r = dict(prompt_id=pid, prompt=prompt, wall_s=wall, other_s=other, phases=phases, clocks=smi.stats)
        results.append(r)
        print(f"[{pid}] {wall:.1f} s | " + " | ".join(
            f"{k} {v['total_s']:.1f} s x{v['calls']} peak {v['peak_gb']:.1f} GB" for k, v in phases.items())
            + f" | other {other:.1f} s | SM {smi.stats.get('sm_mhz_mean', 0):.0f} MHz")
        u8 = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
        if not args.no_save_frames:
            np.save(os.path.join(args.out, f"frames_{pid}.npy"), u8)
        export_to_video(list(frames), os.path.join(args.out, f"video_{pid}.mp4"), fps=16)

    report = dict(
        config=vars(args), weights_gb=weights_gb, weights_with_text_encoder_gb=weights_all_gb,
        text_encoding=encode, warmup_s=warmup_s, runs=results,
        env=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                 diffusers=diffusers.__version__, transformers=transformers.__version__,
                 sdpa_flash=torch.backends.cuda.flash_sdp_enabled(),
                 transformer_dtype=str(pipe.transformer.dtype), tf32=torch.backends.cuda.matmul.allow_tf32,
                 sageattention=_version("sageattention")))
    if args.kernels and ph.captured:
        a, k = ph.captured
        report["kernels"] = profile_forward(ph.orig["transformer"], a, k,
                                            os.path.join(args.out, "transformer_forward.trace.json"))
        kr = report["kernels"]
        print(f"one forward {kr['gpu_ms']:.0f} ms GPU: " +
              ", ".join(f"{c} {s:.0%}" for c, s in sorted(kr["share"].items(), key=lambda x: -x[1])))
        for name, ms, cat in kr["top_kernels"]:
            print(f"  {ms:8.1f} ms  {cat:9s}  {name}")
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
