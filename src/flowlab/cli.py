"""Command-line entry point. See each module for the operations it invokes."""
import argparse
import json
from pathlib import Path
from .config import TrainConfig
from .runtime import write_json


def parser():
    p = argparse.ArgumentParser(description="2D Flow Matching educational lab (CPU default, float32)")
    sub = p.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight", help="Inspect runtime and execute float32 smoke")
    check.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    check.add_argument("--expect-v100", action="store_true")
    check.add_argument("--out", type=Path)
    train = sub.add_parser("train", help="Train velocity weights and save a checkpoint")
    train.add_argument("--config", type=Path, default=Path("configs/moons.json"))
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train.add_argument("--no-plots", action="store_true")
    for name in ("sample", "compare"):
        run = sub.add_parser(name, help="Sample fixed weights" if name == "sample" else "Compare solvers under fixed inputs")
        run.add_argument("--checkpoint", type=Path, required=True)
        run.add_argument("--out", type=Path, required=True)
        run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        run.add_argument("--seed", type=int, default=123)
        run.add_argument("--samples", type=int, default=2048)
        run.add_argument("--no-plots", action="store_true")
        if name == "sample":
            run.add_argument("--steps", type=int, default=32)
            run.add_argument("--method", choices=["euler", "midpoint"], default="midpoint")
        else:
            run.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    try:
        if args.command == "preflight":
            from .preflight import preflight
            result = preflight(args.device, args.expect_v100)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.out, result)
        elif args.command == "train":
            from .train import train
            result = train(TrainConfig.load(args.config), args.out, args.device, not args.no_plots)
        else:
            from .sample import sample, compare
            kw = dict(checkpoint=args.checkpoint, out=args.out, device=args.device, seed=args.seed, samples=args.samples, steps=args.steps, plots=not args.no_plots)
            if args.command == "sample":
                kw["method"] = args.method
                result = sample(**kw)
            else:
                result = compare(**kw)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        p.exit(2, f"flowlab: {error}\n")
