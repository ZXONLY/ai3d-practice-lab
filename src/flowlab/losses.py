"""Training objective: average squared error over BOTH samples and coordinates."""
import torch


def velocity_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("velocity tensors must both have shape (batch,2)")
    return (predicted - target).square().mean()
