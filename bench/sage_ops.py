"""SageAttention as one opaque torch.library custom op, so fullgraph torch.compile can keep it.

sageattention's Python dispatch calls torch.cuda.device_count() and torch.cuda.set_device(),
which Dynamo cannot trace ("torch.* op returned non-Tensor"). Without this wrapper the only way
to compile is fullgraph=False, which splits every block at attention and changes the compile
setup together with the attention kernel. Inside the op, sageattention runs eagerly as it would
without compile; Inductor sees one call with a fake (shape-only) implementation.

install() replaces the sageattention functions that diffusers' attention_dispatch imported
(sageattn, sageattn_qk_int8_pv_fp8_cuda, sageattn_qk_int8_pv_fp16_cuda) with wrappers around
the op. diffusers 0.42.1 calls all three with q, k, v, tensor_layout="NHD", is_causal, sm_scale,
return_lse.
"""
from typing import Optional

import torch

_FNS = {}


@torch.library.custom_op("videodit::sage_attn", mutates_args=())
def sage_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, variant: str, is_causal: bool,
              sm_scale: Optional[float]) -> torch.Tensor:
    out = _FNS[variant](q, k, v, tensor_layout="NHD", is_causal=is_causal, sm_scale=sm_scale)
    return out.contiguous()


@sage_attn.register_fake
def _(q, k, v, variant, is_causal, sm_scale):
    return q.new_empty(q.shape)  # contiguous, like the real output after .contiguous()


def _wrapper(variant):
    def fn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=None, return_lse=False, **kw):
        assert tensor_layout == "NHD" and not return_lse and not kw, (tensor_layout, return_lse, kw)
        return torch.ops.videodit.sage_attn(q, k, v, variant, is_causal, sm_scale)
    return fn


def install():
    import sageattention
    import diffusers.models.attention_dispatch as ad
    _FNS.update(auto=sageattention.sageattn,
                fp8pv=sageattention.sageattn_qk_int8_pv_fp8_cuda,
                fp16pv=sageattention.sageattn_qk_int8_pv_fp16_cuda)
    ad.sageattn = _wrapper("auto")
    ad.sageattn_qk_int8_pv_fp8_cuda = _wrapper("fp8pv")
    ad.sageattn_qk_int8_pv_fp16_cuda = _wrapper("fp16pv")


if __name__ == "__main__":
    # Self-test: eager sageattn vs the op, plain and under fullgraph compile (needs a GPU)
    install()
    q, k, v = (torch.randn(1, 4096, 12, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    ref = _FNS["auto"](q, k, v, tensor_layout="NHD", is_causal=False)
    op = torch.ops.videodit.sage_attn(q, k, v, "auto", False, None)
    comp = torch.compile(lambda a, b, c: torch.ops.videodit.sage_attn(a, b, c, "auto", False, None) * 1.0,
                         fullgraph=True)(q, k, v)
    d_op, d_comp = ((x.float() - ref.float()).abs().max().item() for x in (op, comp))
    print(f"max |op - eager| = {d_op:.3g}, max |compiled - eager| = {d_comp:.3g} (0 expected)")
    assert d_op < 1e-2 and d_comp < 1e-2, "the custom op does not reproduce eager sageattn"
