"""Small end-to-end acceptance tests; not a model-quality benchmark."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from flowlab.checkpoint import load_checkpoint
from flowlab.config import TrainConfig
from flowlab.data import generator, sample_base
from flowlab.evaluation import fixed_distribution_inputs, validation_batch
from flowlab.sample import compare, sample
from flowlab.solvers import integrate
from flowlab.train import train

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.config = TrainConfig.load(ROOT / "configs/smoke.json")
        cls.run_dir = cls.root / "trained"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.summary = train(cls.config, cls.run_dir, plots=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_checkpoint_reload_and_sampling_does_not_train(self):
        a, config, payload = load_checkpoint(self.run_dir / "checkpoint.pt")
        b, _, _ = load_checkpoint(self.run_dir / "checkpoint.pt")
        self.assertEqual(payload["step"], self.config.steps)
        self.assertIn("optimizer_state", payload)
        self.assertIn("train_rng_state", payload)
        original = {k: v.clone() for k,v in a.state_dict().items()}
        z0 = sample_base(16, generator(7))
        ra = integrate(a, z0, 4, "midpoint")
        rb = integrate(b, z0, 4, "midpoint")
        self.assertTrue(torch.equal(ra.final, rb.final))
        for name, parameter in a.state_dict().items():
            self.assertTrue(torch.equal(parameter, original[name]))
        self.assertTrue(all(p.grad is None for p in a.parameters()))

    def test_seeded_training_repeats_exactly_on_cpu(self):
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.config, self.root / "repeated", plots=False)
        a, _, _ = load_checkpoint(self.run_dir / "checkpoint.pt")
        b, _, _ = load_checkpoint(self.root / "repeated/checkpoint.pt")
        for name, parameter in a.state_dict().items():
            self.assertTrue(torch.equal(parameter, b.state_dict()[name]))

    def test_sample_and_comparison_share_identical_inputs(self):
        sample(self.run_dir / "checkpoint.pt", self.root / "sample", seed=999, samples=64, steps=4, plots=False)
        rows = compare(self.run_dir / "checkpoint.pt", self.root / "comparison", seed=999, samples=64, steps=[2,4], plots=False)
        a = torch.load(self.root / "sample/inputs.pt", weights_only=True)
        b = torch.load(self.root / "comparison/inputs.pt", weights_only=True)
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))
        self.assertEqual([r["nfe"] for r in rows], [2,4,4,8])
        self.assertTrue(all(r["sliced_wasserstein_2"] >= 0 for r in rows))

    def test_validation_is_fixed_and_separate(self):
        a, b = validation_batch(self.config, "cpu"), validation_batch(self.config, "cpu")
        for x,y in zip(a,b): self.assertTrue(torch.equal(x,y))
        x, _, _ = fixed_distribution_inputs(64, self.config, 3)
        y, _, _ = fixed_distribution_inputs(64, self.config, 4)
        self.assertFalse(torch.equal(x,y))

    def test_preserves_existing_runs(self):
        with self.assertRaises(ValueError):
            train(self.config, self.run_dir, plots=False)

    def test_cli_preflight_train_sample_compare_and_headless_plots(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        base = [sys.executable, "-m", "flowlab"]
        commands = [
            ["preflight", "--out", str(self.root / "preflight.json")],
            ["train", "--config", str(ROOT / "configs/smoke.json"), "--out", str(self.root / "cli_train")],
            ["sample", "--checkpoint", str(self.root / "cli_train/checkpoint.pt"), "--out", str(self.root / "cli_sample"), "--steps", "2", "--samples", "32", "--no-plots"],
            ["compare", "--checkpoint", str(self.root / "cli_train/checkpoint.pt"), "--out", str(self.root / "cli_compare"), "--steps", "2", "4", "--samples", "32", "--no-plots"],
        ]
        for command in commands:
            with self.subTest(command=command[0]):
                result = subprocess.run(base + command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ("loss.png", "samples.png", "trajectories.png", "vector_field.png"):
            self.assertGreater((self.root / "cli_train" / name).stat().st_size, 1000)
        self.assertEqual(json.loads((self.root / "preflight.json").read_text())["float32_forward_backward"], "passed")

    def test_explicit_unavailable_cuda_fails_without_fallback(self):
        if torch.cuda.is_available(): self.skipTest("CPU-only behavior check")
        result = subprocess.run([sys.executable, "-m", "flowlab", "preflight", "--device", "cuda"], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no CPU fallback", result.stderr)

    def test_notebook_code_cells_execute_in_order(self):
        notebook = json.loads((ROOT / "notebooks/01_flow_matching_by_hand.ipynb").read_text())
        scope = {"__name__": "__main__"}
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "notebook", "exec"), scope)
        self.assertIn("model", scope)

    def test_comparison_uses_shared_plot_limits(self):
        from unittest.mock import patch
        with patch("flowlab.plotting.distributions") as render:
            compare(self.run_dir / "checkpoint.pt", self.root / "common_axes", samples=32, steps=[2,4], plots=True)
        self.assertEqual(len(render.call_args_list), 4)
        limits = [call.kwargs["limits"] for call in render.call_args_list]
        self.assertTrue(all(value == limits[0] for value in limits))
        for call in render.call_args_list:
            for tensor in call.args[:2]:
                self.assertTrue((tensor[:,0] >= limits[0][0][0]).all())
                self.assertTrue((tensor[:,0] <= limits[0][0][1]).all())
                self.assertTrue((tensor[:,1] >= limits[0][1][0]).all())
                self.assertTrue((tensor[:,1] <= limits[0][1][1]).all())

    def test_cpu_preflight_survives_driver_diagnostic_timeout(self):
        from unittest.mock import patch
        from flowlab.preflight import preflight
        with patch("flowlab.preflight.shutil.which", return_value="nvidia-smi"), patch("flowlab.preflight.subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 15)), patch("flowlab.runtime.git_info", return_value={"commit": None, "dirty": None}):
            result = preflight("cpu")
        self.assertEqual(result["float32_forward_backward"], "passed")
        self.assertEqual(result["nvidia_smi"], "diagnostic failed: TimeoutExpired")
