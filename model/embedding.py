import math
import torch
import torch.nn as nn
from einops import rearrange


### Patch Embedding
### ### ### ### ### ### ### ### ### 

class PatchEmbedding(nn.Module):

    def __init__(self, 
        in_height,
        in_width,
        in_channels,
        patch_size,
        hidden_size,
    ):
        super().__init__()
        self.patch_size = patch_size

        patch_dim = in_channels * (patch_size ** 2)
        
        # initialize linear layer to project to hidden dimension
        self.patch_embed = nn.Linear(patch_dim, hidden_size)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

        # precompute positional embeddings
        self.pos_emb = nn.Parameter(
            self.position_embedding(in_height, in_width, patch_size, hidden_size),
            requires_grad=False
        )

    @staticmethod
    def position_embedding(in_height, in_width, patch_size, dim, max_period=10000):
        # determine height/width of grid
        grid_size_h = in_height // patch_size
        grid_size_w = in_width // patch_size
        
        # create grid
        grid_h = torch.arange(grid_size_h, dtype=torch.float32)
        grid_w = torch.arange(grid_size_w, dtype=torch.float32)
        grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0) # (2, h, w)
        
        grid_h_positions = grid[0].reshape(-1) # (h*w)
        grid_w_positions = grid[1].reshape(-1) # (h*w)

        freqs = max_period ** (
            -torch.arange(
                start=0, 
                end=dim // 4, 
                dtype=torch.float32,
            ) / (dim // 4)
        )   # (dim // 4)

        grid_h_emb = torch.einsum("i,j->ij", grid_h_positions, freqs)
        grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)  # (h*w, dim//2)

        grid_w_emb = torch.einsum("i,j->ij", grid_w_positions, freqs)
        grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)  # (h*w, dim//2)

        grid_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)  # (h*w, dim)

        return grid_emb.unsqueeze(0)  # (1, h*w, dim)

    def patchify(self, x):
        x = rearrange(x, "b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)", ph=self.patch_size, pw=self.patch_size)
        return x

    def unpatchify(self, x):
        nh = nw = int(x.shape[1] ** 0.5)
        x = rearrange(x, "b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)", nh=nh, nw=nw, ph=self.patch_size, pw=self.patch_size)
        return x
    
    def forward(self, x):
        # patchify
        x = self.patchify(x)
        # embed to hidden dimension
        x = self.patch_embed(x)
        # incorporate positional embeddings
        x += self.pos_emb
        return x


### Timestep Embedding
### ### ### ### ### ### ### ### ### 

class TimestepEmbedding(nn.Module):

    def __init__(self, 
        freq_embedding_size,
        hidden_size
    ):
        super().__init__()
        self.freq_embedding_size = freq_embedding_size

        # timestep embedding MLP / initialization
        self.mlp = nn.Sequential(
            nn.Linear(freq_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.freq_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


### Class Embedding
### ### ### ### ### ### ### ### ### 

class ClassEmbedding(nn.Module):

    def __init__(self, 
        num_classes,
        hidden_size, 
        dropout_prob=0.1
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        use_cfg_embed = self.use_dropout = dropout_prob > 0

        # embedding table / initialization
        self.embed_table = nn.Embedding(self.num_classes + use_cfg_embed, hidden_size)
        nn.init.normal_(self.embed_table.weight, std=0.02)

    def token_drop(self, labels, force_drop_ids=None):
        """ Drop Label to enable CFG """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeds = self.embed_table(labels)
        return embeds
