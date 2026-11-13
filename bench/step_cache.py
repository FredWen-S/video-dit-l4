"""First-block cache (FBCache) for the Wan transformer that keeps the blocks compiled.

The rule follows diffusers' FirstBlockCache (hooks/first_block_cache.py, 0.42.1):
- Every forward runs block 0. Its residual r = block0(h) - h is compared with the residual saved
  at the last *computed* forward of the same CFG branch:
      rel = mean|r - r_saved| / mean|r_saved|
- If rel < threshold, blocks 1..29 are skipped. The output is block0(h) plus the cached residual
  of blocks 1..29 from the last computed forward.
- Otherwise all blocks run, and r and the blocks-1..29 residual are saved.

Why not diffusers' own hooks: they wrap the block's forward, and FirstBlockCache's decision calls
a @torch.compiler.disable function inside the first block, which breaks fullgraph compile. MagCache
disables compile on every block and needs per-model calibration. Here the wrappers run eagerly
around the compiled blocks, so the compiled graphs are the same as without the cache. The decision
costs one .item() sync per forward.

CFG branches: WanPipeline calls the transformer twice per step, conditional first, then
unconditional (pipeline_wan.py, 0.42.1). Branch = call index mod 2; reset() before every pipeline
call. Residuals are kept in fp32, so a skipped forward adds the cache without an extra bf16
rounding of the residual.
"""
import torch


class _Wrap(torch.nn.Module):
    def __init__(self, cache, block, idx, n):
        super().__init__()
        self.cache, self.block, self.idx, self.n = cache, block, idx, n

    def forward(self, hidden_states, *a, **k):
        c = self.cache
        if self.idx == 0:
            return c._head(self.block, hidden_states, a, k)
        if c.skip:
            out = hidden_states
        else:
            out = self.block(hidden_states, *a, **k)
            if self.idx == self.n - 1:
                c.state[c.branch]["tail"] = out.float() - c.h1
        if self.idx == self.n - 1:
            c.calls += 1
        return out


class FirstBlockCache:
    def __init__(self, transformer, threshold, branches=2):
        self.threshold, self.branches = threshold, branches
        blocks = transformer.blocks
        n = len(blocks)
        for i in range(n):
            blocks[i] = _Wrap(self, blocks[i], i, n)
        self.reset()

    def reset(self):
        self.calls, self.skip, self.branch, self.h1 = 0, False, 0, None
        self.state = {b: {} for b in range(self.branches)}
        self.log = []   # per forward: [step, branch, rel or None, computed]

    def _head(self, block, h, a, k):
        self.branch = self.calls % self.branches
        st = self.state[self.branch]
        h1 = block(h, *a, **k)
        r = h1.float() - h.float()
        rel = None
        if "r" in st and "tail" in st:
            rel = ((r - st["r"]).abs().mean() / st["r"].abs().mean()).item()
        self.skip = rel is not None and rel < self.threshold
        self.log.append([self.calls // self.branches, self.branch, rel, not self.skip])
        if self.skip:
            return (h1.float() + st["tail"]).to(h1.dtype)
        st["r"] = r
        self.h1 = h1.float()
        return h1

    def summary(self):
        computed = sum(1 for x in self.log if x[3])
        return dict(threshold=self.threshold, forwards=len(self.log), computed=computed,
                    skipped=len(self.log) - computed,
                    skipped_steps={b: [x[0] for x in self.log if x[1] == b and not x[3]]
                                   for b in range(self.branches)},
                    log=self.log)
