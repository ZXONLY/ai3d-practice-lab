"""Explicit ODE solvers. Sampling changes z, never optimizer/model parameters."""
from dataclasses import dataclass
import torch

@dataclass
class SolveResult:
    final: torch.Tensor
    trajectory: torch.Tensor  # (steps+1,batch,2); retained for learning/plots
    times: torch.Tensor
    nfe: int


@torch.no_grad()
def integrate(field, z0: torch.Tensor, steps: int = 32, method: str = "euler") -> SolveResult:
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if method not in {"euler", "midpoint"}:
        raise ValueError("method must be euler or midpoint")
    if z0.ndim != 2 or z0.shape[-1] != 2:
        raise ValueError("z0 must have shape (batch,2)")
    z = z0.clone()
    trajectory = [z.clone()]
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((len(z), 1), k * dt, device=z.device, dtype=z.dtype)
        velocity = field(z, t)
        if method == "midpoint":
            velocity = field(z + 0.5 * dt * velocity, t + 0.5 * dt)
        z = z + dt * velocity
        trajectory.append(z.clone())
    return SolveResult(z, torch.stack(trajectory), torch.linspace(0, 1, steps + 1, device=z.device), steps * (2 if method == "midpoint" else 1))
