"""Configuration is data: reject misspelled or invalid experiment parameters."""
from dataclasses import asdict, dataclass, fields
import json
import math
from collections.abc import Mapping
from pathlib import Path

@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "moons"
    seed: int = 42
    steps: int = 2000
    batch_size: int = 256
    learning_rate: float = 0.001
    hidden_dim: int = 128
    hidden_layers: int = 3
    train_size: int = 8192
    val_size: int = 2048
    eval_every: int = 100
    noise: float = 0.08
    threads: int = 1

    def __post_init__(self):
        if not isinstance(self.dataset, str) or self.dataset not in {"moons", "gmm8"}:
            raise ValueError("dataset must be moons or gmm8")
        for name in ("steps", "batch_size", "hidden_dim", "hidden_layers", "train_size", "val_size", "eval_every", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("learning_rate", "noise"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be in (0, 1)")
        if not 0 <= self.noise < 1:
            raise ValueError("noise must be in [0, 1)")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise ValueError("Config must be a JSON object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
