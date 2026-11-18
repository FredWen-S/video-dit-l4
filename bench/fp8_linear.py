"""W8A8 FP8 (e4m3) linear layers on torch._scaled_mm, for the Wan transformer blocks.

Weights are quantized once at install time; activations are quantized on the fly in forward,
which torch.compile fuses into Triton kernels (amax, scale, cast). Two scale granularities:
- tensorwise: one scale for the whole activation and one for the whole weight
- rowwise:    one scale per token (activation row) and one per output channel (weight row)
The GEMM accumulates in fp32 unless fast_accum=True (cuBLASLt's faster, less precise FP8
accumulation). Output is bf16; the bias is added after the GEMM.

Why not torchao: no torchao release lists torch 2.14, and on sm_89 its float8 inference path
ends up in torch._scaled_mm anyway (its faster kernels are gated to sm_90+).

install() must run before compiling, so the compiled graphs contain the quantization.
"""
import torch
import torch.nn.functional as F

E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max   # 448


def quantize(x, rowwise):
    """bf16/fp32 (M, K) -> e4m3 (M, K) and fp32 scale, (M, 1) if rowwise else scalar."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True) if rowwise else xf.abs().amax()
    scale = amax.clamp(min=1e-12) / E4M3_MAX
    q = (xf / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def scaled_mm(xq, xs, wq, ws, fast_accum=False):
    """xq (M, K) e4m3 row-major, wq (N, K) e4m3 row-major -> (M, N) bf16."""
    return torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16,
                            use_fast_accum=fast_accum)


class FP8Linear(torch.nn.Module):
    def __init__(self, lin, rowwise=True, fast_accum=False):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.rowwise, self.fast_accum = rowwise, fast_accum
        with torch.no_grad():
            wq, ws = quantize(lin.weight, rowwise)
        self.register_buffer("wq", wq)
        self.register_buffer("ws", ws.reshape(1, -1).contiguous() if rowwise else ws)  # (1, N) or scalar
        self.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach(), requires_grad=False)

    def forward(self, x):
        shape = x.shape
        xq, xs = quantize(x.reshape(-1, self.in_features), self.rowwise)
        out = scaled_mm(xq, xs, self.wq, self.ws, self.fast_accum)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)

    def extra_repr(self):
        return f"{self.in_features}->{self.out_features}, {'rowwise' if self.rowwise else 'tensorwise'}" \
               f"{', fast_accum' if self.fast_accum else ''}"


def install(blocks, rowwise=True, fast_accum=False):
    """Replaces every nn.Linear inside `blocks` (the transformer's block list) whose dimensions are
    multiples of 16 (a _scaled_mm requirement). Returns the number replaced."""
    n = 0
    for parent in list(blocks.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear) and child.in_features % 16 == 0 and child.out_features % 16 == 0:
                setattr(parent, name, FP8Linear(child, rowwise, fast_accum))
                n += 1
    return n


def reference(x, w):
    return x.float() @ w.float().t()


if __name__ == "__main__":
    # Quick correctness check on one Wan shape (needs a GPU with FP8, sm_89+)
    torch.manual_seed(0)
    x = torch.randn(4096, 1536, device="cuda", dtype=torch.bfloat16)
    lin = torch.nn.Linear(1536, 8960, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x.float(), lin.weight.float(), lin.bias.float())
    for rowwise in (False, True):
        y = FP8Linear(lin, rowwise)(x).float()
        print(f"{'rowwise' if rowwise else 'tensorwise'}: rel L2 err {((y - ref).norm() / ref.norm()).item():.3e}")
