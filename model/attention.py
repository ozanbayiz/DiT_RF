import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Attention(nn.Module):
    
    def __init__(self,
        dim, 
        num_heads 
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # QKV projection layer / initialization
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=True)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)

        # layer norm for Q & K
        self.q_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1E-6)
        self.k_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1E-6)

        # output projection layer / initialization
        self.out_proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)


    def forward(self, x):

        # compute into Q, K, V for each head, then split into Q, K, V
        q, k, v = self.qkv_proj(x).split(self.dim, dim=-1)  # 3 x (dim, dim)

        # layer norm for Q & K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # rearrange Q, K, V into (batch, n_heads, n_patches, d_head)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # compute attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        x = rearrange(x, "b h n d -> b n (h d)")

        # project attention output into hidden dimension
        x = self.out_proj(x)

        return x

