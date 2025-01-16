import torch.nn as nn
from .attention import Attention


def modulate(x, shift, scale):
    return x * (1 + shift.unsqueeze(1)) + scale.unsqueeze(1)


### DiT Block
### ### ### ### ### ### ### ### ### 

class DiTBlock(nn.Module):

    def __init__(self,
        hidden_size,
        num_heads,
        mlp_ratio
    ):
        super().__init__()

        # layer normalization before attention block
        self.attn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1E-6)

        # attention block
        self.attn_block = Attention(
            dim=hidden_size,
            num_heads=num_heads
        )

        # layer normalization before MLP
        self.mlp_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1E-6)

        # MLP / initialization
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_size, hidden_size),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

        # AdaLN-Zero MLP / initialization
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_mod[1].weight)
        nn.init.zeros_(self.adaLN_mod[1].bias)
    

    def forward(self, x, c):
        # get scale/shift parameters
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.adaLN_mod(c).chunk(6, dim=1)

        # apply attention block
        attn_norm_out = modulate(self.attn_norm(x), gamma_1, beta_1)
        x = x + alpha_1.unsqueeze(1) * self.attn_block(attn_norm_out)

        # apply MLP
        mlp_norm_out = modulate(self.mlp_norm(x), gamma_2, beta_2)
        x = x + alpha_2.unsqueeze(1) * self.mlp(mlp_norm_out)

        return x


### Final Layer of DiT
### ### ### ### ### ### ### ### ### 

class FinalLayer(nn.Module):

    def __init__(self, 
        hidden_size, 
        patch_size, 
        in_channels
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True)

        # Adaptive Layer Normalization / initialization
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_mod[1].weight)
        nn.init.zeros_(self.adaLN_mod[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_mod(c).chunk(2, dim=1)
        x = modulate(self.layer_norm(x), shift, scale)
        x = self.linear(x)
        return x      