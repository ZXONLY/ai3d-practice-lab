"""Inference-only experiments: a fixed checkpoint and explicitly saved initial noise."""
from pathlib import Path
import time
import torch
from .checkpoint import load_checkpoint, sha256
from .evaluation import fixed_distribution_inputs, sliced_wasserstein_2
from .runtime import environment, setup, write_json
from .solvers import integrate
from .train import fresh_output


def _load(checkpoint, out, device, seed, samples):
    if samples < 2:
        raise ValueError("samples must be at least 2")
    out = fresh_output(out)
    device = setup(seed, 1, device)
    model, config, payload = load_checkpoint(checkpoint, device)
    z0, target, directions = fixed_distribution_inputs(samples, config, seed)
    inputs = {"initial_noise": z0, "target": target, "projection_directions": directions}
    torch.save(inputs, out / "inputs.pt")
    provenance = {"checkpoint_sha256": sha256(checkpoint), "checkpoint_training_step": payload["step"], "checkpoint_metadata": payload["metadata"], "seed": seed, "samples": samples, "environment": environment(device)}
    write_json(out / "metadata.json", provenance)
    return out, device, model, z0, target, directions


def sample(checkpoint, out, device="cpu", seed=123, samples=2048, steps=32, method="midpoint", plots=True):
    out, device, model, z0, target, directions = _load(checkpoint, out, device, seed, samples)
    result = integrate(model, z0.to(device), steps, method)
    torch.save({"final": result.final.cpu(), "trajectory": result.trajectory.cpu(), "times": result.times.cpu()}, out / "generated.pt")
    metrics = {"method": method, "steps": steps, "nfe": result.nfe, "sliced_wasserstein_2": sliced_wasserstein_2(result.final.cpu(), target, directions)}
    write_json(out / "summary.json", metrics)
    if plots:
        from . import plotting
        plotting.distributions(target, result.final, out / "samples.png")
        plotting.trajectories(result, target, out / "trajectories.png")
        plotting.vector_field(model, device, out / "vector_field.png")
    return metrics


def compare(checkpoint, out, device="cpu", seed=123, samples=2048, steps=(4, 8, 16, 32, 64), plots=True):
    if not steps or any(type(s) is not int or s < 1 for s in steps):
        raise ValueError("All solver step counts must be positive integers")
    out, device, model, z0, target, directions = _load(checkpoint, out, device, seed, samples)
    initial = z0.to(device)
    rows, generated_clouds = [], []
    for method in ("euler", "midpoint"):
        for count in steps:
            # No retraining or regeneration of z0/target/projection directions here.
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = integrate(model, initial, count, method)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            row = {"method": method, "steps": count, "nfe": result.nfe, "sliced_wasserstein_2": sliced_wasserstein_2(result.final.cpu(), target, directions), "seconds_single_run": elapsed}
            rows.append(row)
            if plots:
                generated_clouds.append(result.final.cpu())
    if plots:
        from . import plotting
        shared_limits = plotting.distribution_limits(target, *generated_clouds)
        for row, generated in zip(rows, generated_clouds):
            method, count, nfe = row["method"], row["steps"], row["nfe"]
            plotting.distributions(target, generated, out / f"samples_{method}_{count}.png", title=f"{method}, {count} steps / {nfe} NFE", limits=shared_limits)
    write_json(out / "comparison.json", {"results": rows, "control": "same checkpoint, z0, target, projections; see metadata.json and inputs.pt", "timing_caution": "Single un-warmed timing; not a throughput benchmark", "metric_caution": "Finite samples, one seed; distribution diagnostic, not exact W2"})
    if plots:
        from . import plotting
        plotting.solver_comparison(rows, out / "comparison.png")
    return rows
