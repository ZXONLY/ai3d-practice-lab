"""Headless by design: save files on SSH servers; never open a GUI window."""
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "flowlab-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

COLORS = {"target": "#27838b", "generated": "#e47746"}


def _axes(ax, title):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def distribution_limits(*clouds):
    values = np.concatenate([cloud.detach().cpu().numpy() for cloud in clouds])
    low, high = values.min(axis=0) - .2, values.max(axis=0) + .2
    return ((float(low[0]), float(high[0])), (float(low[1]), float(high[1])))


def distributions(target, generated, path, title="Generated distribution", limits=None):
    target, generated = target.cpu().numpy(), generated.cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, values, name, color in [(axes[0], target, "Held-out target", COLORS["target"]), (axes[1], generated, title, COLORS["generated"])]:
        ax.scatter(*values.T, s=3, alpha=0.4, color=color, rasterized=True)
        _axes(ax, name)
    axes[2].scatter(*target.T, s=3, alpha=0.25, color=COLORS["target"], label="target")
    axes[2].scatter(*generated.T, s=3, alpha=0.25, color=COLORS["generated"], label="generated")
    _axes(axes[2], "Overlay")
    axes[2].legend(markerscale=3)
    if limits is None:
        both = np.concatenate([target, generated])
        low, high = both.min(axis=0) - .2, both.max(axis=0) + .2
        limits = ((low[0], high[0]), (low[1], high[1]))
    for ax in axes:
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def trajectories(result, target, path, count=40):
    values = result.trajectory[:, :count].cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(*target.cpu().numpy().T, s=3, alpha=.1, color=COLORS["target"], label="target")
    for i in range(values.shape[1]):
        ax.plot(values[:, i, 0], values[:, i, 1], linewidth=.8, alpha=.65)
    ax.scatter(*values[0].T, s=16, marker="x", color="#334155", label="t=0 Gaussian")
    ax.scatter(*values[-1].T, s=16, color=COLORS["generated"], label="t=1 generated")
    _axes(ax, f"Learned ODE trajectories ({result.nfe} field evaluations)")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


@torch.no_grad()
def vector_field(model, device, path):
    xs = torch.linspace(-3, 3, 17, device=device)
    xx, yy = torch.meshgrid(xs, xs, indexing="xy")
    grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, time in zip(axes, (0., .5, 1.)):
        t = torch.full((len(grid), 1), time, device=device)
        velocity = model(grid, t).cpu().numpy()
        ax.quiver(grid[:, 0].cpu(), grid[:, 1].cpu(), velocity[:, 0], velocity[:, 1], angles="xy", scale_units="xy", scale=6, width=.004)
        _axes(ax, f"Learned velocity at t={time:g} (arrows / 6)")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def loss_curve(rows, path):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    training = [r for r in rows if r["train_loss"] is not None]
    ax.plot([r["step"] for r in training], [r["train_loss"] for r in training], label="train interval mean", color=COLORS["target"])
    ax.plot([r["step"] for r in rows], [r["val_loss"] for r in rows], label="fixed held-out pairs", color=COLORS["generated"])
    ax.set(xlabel="optimizer steps", ylabel="velocity MSE", title="Regression loss (not a generation quality score)")
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def solver_comparison(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method in ("euler", "midpoint"):
        r = [x for x in rows if x["method"] == method]
        for ax, key, label in zip(axes, ("steps", "nfe"), ("ODE steps", "Network field evaluations (NFE)")):
            ax.plot([x[key] for x in r], [x["sliced_wasserstein_2"] for x in r], "o-", label=method)
            ax.set(xlabel=label, ylabel="Empirical sliced W2 (lower is closer)")
            ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Same checkpoint, initial noise, target sample and projections")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
