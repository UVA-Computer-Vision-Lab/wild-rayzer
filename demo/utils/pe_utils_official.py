import torch
import numpy as np


def get_1d_sincos_pos_emb_from_grid(embed_dim, pos, device="cpu"):
    assert embed_dim % 2 == 0
    pos = pos.float() if isinstance(pos, torch.Tensor) else torch.tensor(pos, dtype=torch.float32)
    pos = pos.to(device)
    dim = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
    freq = 1.0 / (10000 ** (dim / (embed_dim // 2)))
    emb_sin = torch.sin(pos[:, None] * freq)
    emb_cos = torch.cos(pos[:, None] * freq)
    return torch.cat([emb_sin, emb_cos], dim=-1)


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token: bool = False,
    extra_tokens: int = 0,
    scale: float = 1.0,
    base_size=None,
    device: str = "cpu",
):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    H, W = grid_size
    grid_h = np.arange(H, dtype=np.float32) / scale
    grid_w = np.arange(W, dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / H
        grid_w *= base_size / W
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, W, H])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device=device)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat([torch.zeros([extra_tokens, embed_dim], device=device), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device="cpu"):
    assert embed_dim % 2 == 0
    if isinstance(grid, np.ndarray):
        grid_t = torch.from_numpy(grid).to(device=device, dtype=torch.float32)
    else:
        grid_t = grid.to(device=device, dtype=torch.float32)
    emb_h = get_1d_sincos_pos_emb_from_grid(embed_dim // 2, grid_t[0].reshape(-1), device)
    emb_w = get_1d_sincos_pos_emb_from_grid(embed_dim // 2, grid_t[1].reshape(-1), device)
    return torch.cat([emb_h, emb_w], dim=1)


def rope(positions: torch.Tensor, d: int, device="cpu") -> torch.Tensor:
    B, N = positions.shape
    half_d = d // 2
    positions_3d = positions.unsqueeze(-1)
    idx = torch.arange(half_d, device=device).view(1, 1, -1)
    freqs = torch.pow(10000.0, -2.0 * idx / d)
    angle = positions_3d.to(device) * freqs
    sin_part = angle.sin()
    cos_part = angle.cos()
    embeddings = torch.empty(B, N, d, device=device, dtype=positions.dtype)
    embeddings[..., 0::2] = sin_part
    embeddings[..., 1::2] = cos_part
    return embeddings


