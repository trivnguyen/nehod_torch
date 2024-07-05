
import tqdm

import torch
import numpy as np

from models import diffusion_utils

# Some helper functions
def create_mask(n_array, num_particles):
    """
    Create a mask of shape (len(n_array), num_particles) where for each row i,
    the first n_array[i] elements are 1, and the rest are 0.
    """
    # Create an array of indices [0, 1, 2, ..., num_particles-1]
    indices = np.arange(num_particles)

    # Use broadcasting to create the mask
    masks = indices < n_array[:, None]

    return masks

# Generator functions
@torch.no_grad()
def generate_batch(
    model, cond=None, mask=None, num_sampling_steps=1000, num_particles=50,
    num_features=9, batch_size=None, device='cuda'):
    """ Generate VDM samples from a single batch """
    if cond is None and batch_size is None:
        raise ValueError("If unconditional, batch_size must be given")
    batch_size = cond.shape[0] if cond is not None else batch_size

    # random noise from N(0, 1)
    z_t = torch.randn((batch_size, num_particles, num_features), device=device)
    for i in torch.arange(0, num_sampling_steps):
        # denoising step
        z_t = model.sample_step(
            z_t, i, num_sampling_steps, conditioning=cond, mask=mask)
    g0 = model.gamma(0.0)
    var0 = diffusion_utils.sigma2(g0)
    z_0_rescaled = z_t / torch.sqrt(1.0 - var0)

    samples = model.decode(z_0_rescaled).loc
    return samples

@torch.no_grad()
def generate_from_loader(
    loader, model, num_sampling_steps=1000, num_particles=50, num_features=9,
    device='cuda', verbose=True, norm_dict=None):
    """ Generate VDM samples from data loader """
    model.to(device)
    model.eval()
    if verbose:
        pbar = tqdm.tqdm(loader, total=len(loader))
    else:
        pbar = loader

    gen_samples = []
    for i, batch in enumerate(pbar):
        cond, mask = [b.to(device) for b in batch]
        samples = generate_batch(
            model, cond, mask, num_sampling_steps, num_particles, num_features)
        gen_samples.append(samples.cpu().numpy())
    gen_samples = np.concatenate(gen_samples, axis=0)

    if norm_dict is not None:
        gen_samples = gen_samples * norm_dict['x_std'] + norm_dict['x_mean']
    return gen_samples