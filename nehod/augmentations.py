
import torch
import numpy as np

def random_symmetry_matrix(device: torch.device = None):
    # 8 possible sign combinations for reflections
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )

    # 6 permutations for axis swapping
    perms = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])

    # Randomly select one sign combination and one permutation
    sign = signs[torch.randint(0, 8, (1,)).item()]
    perm = perms[torch.randint(0, 6, (1,)).item()]

    # Combine them to form the random symmetry matrix
    matrix = np.eye(3)[perm] * sign
    return torch.tensor(matrix, dtype=torch.float32, device=device)

def augment_with_symmetries(
    x, n_pos_dim: int = 3, n_vel_dim: int = 3, device: torch.device = None
):
    """
    Apply random rotations and reflections to the positions and velocities in x.
    The first n_pos_dim dimensions are assumed to be positions, and the next n_vel_dim
    dimensions are assumed to be velocities. The rest of the dimensions are left
    unchanged.
    """

    # Rotations and reflections that respect boundary conditions
    matrix = random_symmetry_matrix(device)
    x[..., :n_pos_dim] = torch.matmul(x[..., :n_pos_dim], matrix.T)
    if n_vel_dim > 0:
        # Rotate velocities too
        x[..., n_pos_dim : n_pos_dim + n_vel_dim] = torch.matmul(
            x[..., n_pos_dim : n_pos_dim + n_vel_dim], matrix.T
        )
    return x
