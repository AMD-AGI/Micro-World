# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.cuda.amp as amp

from .fuser import xFuserLongContextAttention


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply_local(x, freqs):
    dtype = x.dtype
    b, s, n, _ = x.shape
    freqs = freqs[:s].to(device=x.device)
    x_i = torch.view_as_complex(x.to(torch.float32).reshape(b, s, n, -1, 2))
    x_i = torch.view_as_real(x_i * freqs).flatten(-2)
    return x_i.to(dtype)


def usp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(tensor):
        return tensor if tensor.dtype in half_dtypes else tensor.to(dtype)

    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    q = rope_apply_local(q, freqs)
    k = rope_apply_local(k, freqs)

    x = xFuserLongContextAttention()(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size,
    )

    x = x.flatten(2)
    x = self.o(x)
    return x
