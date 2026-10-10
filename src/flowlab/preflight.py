"""A real float32 forward/backward and ODE smoke, not just a CUDA availability flag."""
import shutil
import subprocess
import torch
from .model import VelocityMLP
from .runtime import environment, setup
from .solvers import integrate


def preflight(device="cpu", expect_v100=False):
    device = setup(0, 1, device)
    report = environment(device)
    if expect_v100 and device.type != "cuda":
        raise ValueError("--expect-v100 requires --device cuda")
    if device.type == "cuda" and expect_v100:
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        if "V100" not in name or capability != (7, 0):
            raise RuntimeError(f"Expected V100 (CC 7.0), got {name} (CC {capability})")
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
            report["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        except (subprocess.TimeoutExpired, OSError) as error:
            # Driver diagnostics are supplementary; actual torch operations below
            # are the acceptance test, including for a CPU-only invocation.
            report["nvidia_smi"] = f"diagnostic failed: {type(error).__name__}"
    else:
        report["nvidia_smi"] = "not found"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = VelocityMLP(32, 2).to(device=device, dtype=torch.float32)
    z = torch.randn(16, 2, device=device)
    t = torch.rand(16, 1, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss = model(z, t).square().mean()
    loss.backward(); optimizer.step()
    model.eval()
    result = integrate(model, z, steps=4, method="midpoint")
    if not torch.isfinite(result.final).all():
        raise RuntimeError("Non-finite output in preflight")
    if device.type == "cuda":
        torch.cuda.synchronize()
    report.update(float32_forward_backward="passed", midpoint_sampling="passed", preflight_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    return report
