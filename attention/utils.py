import torch

"""
Rope
"""

import torch

"""
RoPE taken from GPT-NeoX style, via llama in transformers.


"""

def rotate_half(x):
    #x1, x2 = x.chunk(2, dim=-1)
    #return torch.cat((-x2, x1), dim=-1)
    assert x.shape[-1] % 2 == 0, "RoPE requires an even number of dimensions"
    # 偶数维度
    x1 = -x[..., 1::2]
    # 奇数维度
    x2 = x[..., 0::2]
    return torch.stack((x1, x2), dim=-1).reshape(x.shape)



def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def apply_rope_x(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

