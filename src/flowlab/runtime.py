"""Explicit device selection and provenance. CPU is always the default."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import torch


def select_device(name: str) -> torch.device:
    if name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable. Run preflight; no CPU fallback was used.")
    return torch.device(name)


def setup(seed: int, threads: int, device: str):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return select_device(device)


def git_info():
    repo = Path(__file__).resolve().parents[2]
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def environment(device):
    info = {"python": sys.version.split()[0], "torch": str(torch.__version__), "platform": platform.platform(), "device": str(device), "dtype": "float32", "cuda_runtime": torch.version.cuda, "git": git_info()}
    if str(device) == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update(gpu_name=p.name, compute_capability=list(torch.cuda.get_device_capability(device)), gpu_total_memory_bytes=p.total_memory, compiled_arches=torch.cuda.get_arch_list())
    return info


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
