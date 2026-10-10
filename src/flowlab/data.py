"""Pure local float32 data. Each generator is explicit, independent, and CPU-backed."""
import math
import torch


def generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def sample_base(n: int, rng: torch.Generator) -> torch.Tensor:
    if n < 1:
        raise ValueError("n must be positive")
    return torch.randn(n, 2, generator=rng, dtype=torch.float32)


def sample_target(n: int, dataset: str, rng: torch.Generator, noise: float = 0.08) -> torch.Tensor:
    """Generate independent samples; no files, sklearn, or network required.

    moons: two balanced-in-expectation arcs, centered then scaled by 1.5.
    gmm8: eight equally likely centers of radius 2 with isotropic noise.
    Noise is applied after coordinate scaling and is measured in plot units.
    """
    if n < 1 or noise < 0:
        raise ValueError("n must be positive and noise nonnegative")
    if dataset == "moons":
        angle = torch.rand(n, generator=rng) * math.pi
        second = torch.randint(0, 2, (n,), generator=rng).bool()
        x = torch.where(second, 1 - angle.cos(), angle.cos())
        y = torch.where(second, 0.5 - angle.sin(), angle.sin())
        points = (torch.stack((x, y), dim=1) - torch.tensor([0.5, 0.25])) * 1.5
    elif dataset == "gmm8":
        angle = torch.randint(0, 8, (n,), generator=rng) * (2 * math.pi / 8)
        points = 2 * torch.stack((angle.cos(), angle.sin()), dim=1)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return points + noise * torch.randn(n, 2, generator=rng)
