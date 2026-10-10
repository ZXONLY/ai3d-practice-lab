"""Analytic conditional probability path; it has no trainable weights."""
import torch


def straight_path(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor):
    """t=0 noise, t=1 data: z_t=(1-t)z0+t*z1; u_t=dz_t/dt=z1-z0.

    z0,z1 are (batch,2); t is (batch,1). Return two (batch,2) tensors.
    The target velocity is constant for this particular paired straight path.
    """
    if z0.ndim != 2 or z0.shape[-1] != 2 or z1.shape != z0.shape:
        raise ValueError("z0 and z1 must both have shape (batch, 2)")
    if t.shape != (z0.shape[0], 1):
        raise ValueError("t must have shape (batch, 1)")
    if torch.any((t < 0) | (t > 1)):
        raise ValueError("t must be in [0, 1]")
    return (1 - t) * z0 + t * z1, z1 - z0
