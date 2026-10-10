"""Held-out regression loss and an empirical distribution diagnostic are different."""
import torch
from .data import generator, sample_base, sample_target
from .losses import velocity_mse
from .path import straight_path


def validation_batch(config, device):
    rng = generator(config.seed + 10000)
    z1 = sample_target(config.val_size, config.dataset, rng, config.noise)
    z0 = sample_base(config.val_size, rng)
    t = torch.rand(config.val_size, 1, generator=rng)
    zt, target_velocity = straight_path(z0, z1, t)
    return zt.to(device), t.to(device), target_velocity.to(device)


@torch.no_grad()
def heldout_loss(model, batch):
    was_training = model.training
    model.eval()
    value = velocity_mse(model(batch[0], batch[1]), batch[2]).item()
    model.train(was_training)
    return value


def fixed_distribution_inputs(n, config, seed):
    # Distinct generators make adding a target diagnostic unable to change base noise.
    z0 = sample_base(n, generator(seed))
    target = sample_target(n, config.dataset, generator(seed + 1), config.noise)
    directions = torch.randn(128, 2, generator=generator(seed + 2))
    directions = directions / directions.norm(dim=1, keepdim=True)
    return z0, target, directions


def sliced_wasserstein_2(generated, target, directions):
    """sqrt(mean squared sorted projected distances); finite-sample diagnostic.

    Not exact 2D W2, not a likelihood, not a proof of distribution equality.
    Identical sample counts and fixed projection directions are required.
    """
    if generated.shape != target.shape or generated.ndim != 2 or generated.shape[1] != 2:
        raise ValueError("generated and target must have the same (n,2) shape")
    if len(generated) == 0:
        raise ValueError("point clouds must be nonempty")
    if directions.ndim != 2 or directions.shape[1] != 2 or len(directions) == 0:
        raise ValueError("directions must have nonempty shape (projections,2)")
    for name, tensor in (("generated", generated), ("target", target), ("directions", directions)):
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite floating-point values")
    norms = directions.norm(dim=1, keepdim=True)
    if torch.any(norms == 0) or not torch.isfinite(norms).all():
        raise ValueError("projection directions must have finite nonzero length")
    # Normalize here as well as in fixed_distribution_inputs: public API callers
    # must not silently change the distance units by scaling directions.
    directions = directions / norms
    a = (generated @ directions.T).sort(dim=0).values
    b = (target @ directions.T).sort(dim=0).values
    return ((a - b).square().mean()).sqrt().item()
