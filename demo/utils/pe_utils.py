import torch
import math
import numpy as np


def get_1d_sincos_pos_emb_from_grid(embed_dim, pos, device="cpu"):
    """
    Generate 1D sinusoidal positional embeddings from grid positions.

    Args:
        embed_dim (int): The embedding dimension (must be even).
        pos (torch.Tensor): The grid positions (e.g., [0, 1, 2, ..., v-1]).
                           Shape: [b * gh * gw] or [batch_size, sequence_length].
        device (str): Device for the output tensor.

    Returns:
        torch.Tensor: Sinusoidal positional embeddings.
                      Shape: [len(pos), embed_dim]
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even for sine and cosine."

    # Convert positions to float
    pos = pos.float()

    # Compute the sinusoidal frequencies
    dim = torch.arange(
        embed_dim // 2, dtype=torch.float32, device=device
    )  # [0, 1, ..., embed_dim // 2 - 1]
    freq = 1.0 / (10000 ** (dim / (embed_dim // 2)))  # Scale frequencies logarithmically

    # Calculate sine and cosine embeddings
    pos_emb_sin = torch.sin(pos[:, None] * freq)  # Shape: [len(pos), embed_dim // 2]
    pos_emb_cos = torch.cos(pos[:, None] * freq)  # Shape: [len(pos), embed_dim // 2]

    # Concatenate sine and cosine along the last dimension
    pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)  # Shape: [len(pos), embed_dim]

    return pos_emb


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token: bool = False,
    extra_tokens: int = 0,
    scale: float = 1.0,
    base_size=None,
    device: str = "cpu",
):
    """
    Official RAYZAR 2D sine-cosine positional embeddings.

    Args:
        embed_dim: embedding dimension (even)
        grid_size: int or tuple (H, W)
        cls_token: unused here but kept for compatibility
        extra_tokens: if > 0, prepend zero embeddings (compat)
        scale: coordinate scale factor
        base_size: optional base size for scaling
        device: output device

    Returns:
        Tensor of shape [H*W (+extra_tokens), embed_dim]
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    # Build numpy grids exactly like the official implementation (w first)
    H, W = grid_size
    grid_h = np.arange(H, dtype=np.float32) / scale
    grid_w = np.arange(W, dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / H
        grid_w *= base_size / W

    # Note: meshgrid called with (grid_w, grid_h) so w goes first
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    # Match official reshape order [2, 1, W, H] to preserve token ordering
    grid = grid.reshape([2, 1, W, H])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device=device)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat(
            [torch.zeros([extra_tokens, embed_dim], device=device), pos_embed], dim=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device="cpu"):
    """Official helper to build 2D sin-cos embeddings from a (w-first) grid.

    Args:
        embed_dim: even
        grid: numpy array of shape [2, 1, W, H] (w first), as in official code
        device: output device
    Returns:
        Tensor [H*W, embed_dim]
    """
    assert embed_dim % 2 == 0

    # Convert grid to torch tensors on the target device
    if isinstance(grid, np.ndarray):
        grid_t = torch.from_numpy(grid).to(device=device, dtype=torch.float32)
    else:
        grid_t = grid.to(device=device, dtype=torch.float32)

    # In the official implementation emb_h uses grid[0] and emb_w uses grid[1]
    emb_h = get_1d_sincos_pos_emb_from_grid(embed_dim // 2, grid_t[0].reshape(-1), device)
    emb_w = get_1d_sincos_pos_emb_from_grid(embed_dim // 2, grid_t[1].reshape(-1), device)
    pos_embed = torch.cat([emb_h, emb_w], dim=1)
    return pos_embed


def rope(positions: torch.Tensor, d: int, device="cpu") -> torch.Tensor:
    """
    Given a batch of positions in [0,1], compute RoPE-style
    sine-cosine embeddings in dimension d (must be even).

    positions: (B, N) tensor of float positions in [0,1].
    d: int, dimension of the embedding (should be even).
    Returns:
      embeddings: (B, N, d) tensor of float embeddings.
    """
    # positions shape: [B, N]
    B, N = positions.shape
    half_d = d // 2

    # Expand positions to shape [B, N, 1]
    positions_3d = positions.unsqueeze(-1)  # [B, N, 1]

    # Prepare index and frequency tensors
    # idx => [1, 1, half_d]
    idx = torch.arange(half_d, device=device).view(1, 1, -1)
    # freqs => [1, 1, half_d], broadcast to [B, N, half_d]
    freqs = torch.pow(10000.0, -2.0 * idx / d)

    # angle => [B, N, half_d]
    angle = positions_3d.to(device) * freqs

    # Compute sine and cosine => each [B, N, half_d]
    sin_part = angle.sin()
    cos_part = angle.cos()

    # Interleave sine and cosine along the last dimension => [B, N, d]
    embeddings = torch.empty(B, N, d, device=device, dtype=positions.dtype)
    embeddings[..., 0::2] = sin_part
    embeddings[..., 1::2] = cos_part

    return embeddings
