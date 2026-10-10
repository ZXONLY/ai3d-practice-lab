"""Explicit checkpoint format. Only load this lab's trusted, self-produced files."""
import hashlib
from pathlib import Path
import torch
from .config import TrainConfig
from .model import VelocityMLP

FORMAT_VERSION = 1


def save_checkpoint(path, model, optimizer, config, step, metadata, train_rng):
    payload = {
        "format_version": FORMAT_VERSION,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "config": config.to_dict(), "step": step, "metadata": metadata,
        "torch_rng_state": torch.get_rng_state(), "train_rng_state": train_rng.get_state(),
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # weights_only avoids general pickle execution; never accept unknown checkpoint sources.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    config = TrainConfig.from_dict(payload["config"])
    model = VelocityMLP(config.hidden_dim, config.hidden_layers).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
