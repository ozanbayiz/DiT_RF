import torch
import torch.nn as nn
from .embedding import PatchEmbedding, TimestepEmbedding, ClassEmbedding
from .DiT_layers import DiTBlock, FinalLayer

class DiT(nn.Module):
    def __init__(self, 
        in_height,
        in_width,
        in_channels,
        num_classes,
        patch_size,
        time_embedding_size,
        num_layers,
        num_heads,
        hidden_size,
        mlp_ratio=4.0,
        dropout_prob=0.1
    ):
        super().__init__()
        self.num_classes = num_classes

        # embedders
        self.patch_embed = PatchEmbedding(
            in_height=in_height,
            in_width=in_width,
            in_channels=in_channels,
            patch_size=patch_size,
            hidden_size=hidden_size
        )
        self.time_embed = TimestepEmbedding(
            freq_embedding_size=time_embedding_size,
            hidden_size=hidden_size
        )
        self.class_embed = ClassEmbedding(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob
        )

        # transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            ) for _ in range(num_layers)
        ])

        # final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)


    def forward(self, x, t, y):
        # embed image, time, & class
        x = self.patch_embed(x)
        t = self.time_embed(t)
        y = self.class_embed(y, self.training)
        # combine time & class embedding for conditioning
        c = t + y

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # apply final layer
        x = self.final_layer(x, c)

        # patches -> images
        x = self.patch_embed.unpatchify(x)

        return x
    

    def forward_with_cfg(self, x, t, y, cfg_scale=2.0):
        # inspired by https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

        # double input for conditional & unconditional predictions
        dbl_x = torch.cat([x, x], dim=0)
        dbl_t = torch.cat([t, t], dim=0)

        # create labels for conditional & unconditional predictions
        y_uncond = torch.ones_like(y) * self.num_classes # unconditional label
        comb_y = torch.cat([y, y_uncond], dim=0)

        # forward pass
        eps = self.forward(dbl_x, dbl_t, comb_y)

        # split predictions into conditional & unconditional
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        # apply CFG
        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

        return eps