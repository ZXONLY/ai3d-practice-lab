"""The full training loop, with each mathematical operation left visible."""
import json
from pathlib import Path
import time
import torch
from .checkpoint import save_checkpoint
from .data import generator, sample_base, sample_target
from .evaluation import fixed_distribution_inputs, heldout_loss, sliced_wasserstein_2, validation_batch
from .losses import velocity_mse
from .model import VelocityMLP
from .path import straight_path
from .runtime import environment, setup, write_json
from .solvers import integrate


def fresh_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}. Choose a new run name to preserve evidence.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def train(config, out, device="cpu", plots=True):
    out = fresh_output(out)
    device = setup(config.seed, config.threads, device)
    model = VelocityMLP(config.hidden_dim, config.hidden_layers).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    training_data = sample_target(config.train_size, config.dataset, generator(config.seed + 1), config.noise)
    train_rng = generator(config.seed + 2)
    validation = validation_batch(config, device)
    metadata = {"environment": environment(device), "config": config.to_dict(), "parameter_count": sum(p.numel() for p in model.parameters()), "convention": "t=0 Gaussian; t=1 target; independent coupling; float32", "validation_seed": config.seed + 10000, "checkpoint_policy": "final step; not selected by validation"}
    write_json(out / "metadata.json", metadata)
    start = time.perf_counter()
    rows = []
    def log(step, train_loss):
        row = {"step": step, "train_loss": train_loss, "val_loss": heldout_loss(model, validation), "elapsed_seconds": time.perf_counter() - start}
        rows.append(row)
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
    log(0, None)
    interval_loss, interval_count = 0., 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, config.steps + 1):
        indices = torch.randint(config.train_size, (config.batch_size,), generator=train_rng)
        z1 = training_data[indices].to(device)
        z0 = sample_base(config.batch_size, train_rng).to(device)
        t = torch.rand(config.batch_size, 1, generator=train_rng).to(device)
        zt, target_velocity = straight_path(z0, z1, t)
        predicted_velocity = model(zt, t)
        loss = velocity_mse(predicted_velocity, target_velocity)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()  # This is where the persistent weights change.
        interval_loss += loss.item(); interval_count += 1
        if step % config.eval_every == 0 or step == config.steps:
            log(step, interval_loss / interval_count)
            interval_loss, interval_count = 0., 0
    metadata["training_seconds"] = time.perf_counter() - start
    metadata["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    model.eval()
    save_checkpoint(out / "checkpoint.pt", model, optimizer, config, config.steps, metadata, train_rng)
    write_json(out / "metadata.json", metadata)
    z0, target, directions = fixed_distribution_inputs(2048, config, config.seed + 20000)
    sample = integrate(model, z0.to(device), steps=32, method="midpoint")
    score = sliced_wasserstein_2(sample.final.cpu(), target, directions)
    summary = {"initial_val_loss": rows[0]["val_loss"], "final_val_loss": rows[-1]["val_loss"], "sliced_wasserstein_2": score, "sample_method": "midpoint", "sample_steps": 32, "sample_nfe": 64, "sample_seed": config.seed + 20000, "sample_count": 2048, "training_seconds": metadata["training_seconds"], "checkpoint": "checkpoint.pt", "caution": "One seed and finite-sample diagnostics; not a 3D model or a V100 validation."}
    write_json(out / "summary.json", summary)
    if plots:
        from . import plotting
        plotting.loss_curve(rows, out / "loss.png")
        plotting.distributions(target, sample.final, out / "samples.png")
        plotting.trajectories(sample, target, out / "trajectories.png")
        plotting.vector_field(model, device, out / "vector_field.png")
    return summary
