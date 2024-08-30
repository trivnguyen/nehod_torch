
import numpy as np
import torch
import torch.nn.functional as F


def gamma(ts, gamma_min=-6, gamma_max=6):
    return gamma_max + (gamma_min - gamma_max) * ts

def sigma2(gamma):
    return F.sigmoid(-gamma)

def alpha(gamma):
    return torch.sqrt(1 - sigma2(gamma))

def variance_preserving_map(x, gamma, eps):
    """ Apply variance preserving map to x
    Parameters
    ----------
    x : torch.Tensor
        Input tensor, shape (batch_size, n_features)
    gamma : torch.Tensor
        Noise level, assume same shape as x
    eps : torch.Tensor
        Noise tensor, assume same shape as x
    """
    a = alpha(gamma)
    var = sigma2(gamma)
    x_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    eps = eps.reshape(eps.shape[0], -1)
    noise_augmented = a * x + torch.sqrt(var) * eps
    return noise_augmented.reshape(x_shape)

def get_timestep_embedding(
    timesteps, embedding_dim, dtype=torch.float32, device='cuda'):
    """Build sinusoidal embeddings (from Fairseq)."""

    assert len(timesteps.shape) == 1
    timesteps = timesteps * 1000

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10_000, dtype=dtype, device=device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=device) * -emb)
    emb = timesteps.type(dtype)[:, None] * emb[None, :]
    emb = torch.concatenate([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # Zero pad
        emb = F.pad(emb, (0, 1))
        # emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

