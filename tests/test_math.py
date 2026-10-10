import unittest
import torch
from flowlab.data import generator, sample_base, sample_target
from flowlab.path import straight_path
from flowlab.model import VelocityMLP
from flowlab.losses import velocity_mse
from flowlab.solvers import integrate
from flowlab.evaluation import sliced_wasserstein_2
from flowlab.config import TrainConfig

class MathTests(unittest.TestCase):
    def test_hand_calculation(self):
        z0, z1 = torch.tensor([[0., 2.]]), torch.tensor([[4., 0.]])
        zt, velocity = straight_path(z0, z1, torch.tensor([[.25]]))
        torch.testing.assert_close(zt, torch.tensor([[1., 1.5]]))
        torch.testing.assert_close(velocity, torch.tensor([[4., -2.]]))
        self.assertEqual(velocity_mse(torch.tensor([[3., -1.]]), velocity).item(), 1.)

    def test_path_endpoints_and_derivative(self):
        z0, z1 = sample_base(7, generator(1)), sample_base(7, generator(2))
        for t, expected in [(0., z0), (1., z1)]:
            torch.testing.assert_close(straight_path(z0, z1, torch.full((7, 1), t))[0], expected)
        t = torch.full((7, 1), .3, requires_grad=True)
        zt, u = straight_path(z0, z1, t)
        derivative = torch.autograd.grad(zt.sum(), t)[0]
        torch.testing.assert_close(derivative, u.sum(dim=1, keepdim=True))
        with self.assertRaises(ValueError):
            straight_path(z0, z1, torch.ones(7))

    def test_solvers_constant_velocity(self):
        start = sample_base(10, generator(5))
        untouched = start.clone()
        velocity = torch.tensor([4., -2.])
        for method, multiplier in [("euler", 1), ("midpoint", 2)]:
            result = integrate(lambda z, t: velocity.expand_as(z), start, 8, method)
            torch.testing.assert_close(result.final, start + velocity)
            torch.testing.assert_close(result.trajectory[0], start)
            torch.testing.assert_close(start, untouched)
            self.assertEqual(result.trajectory.shape, (9, 10, 2))
            self.assertEqual(result.nfe, 8 * multiplier)

    def test_midpoint_numerical_accuracy(self):
        # dz/dt=z => z(1)=e*z(0); midpoint should have smaller truncation error.
        z0 = torch.ones(1, 2)
        exact = torch.exp(z0)
        euler = integrate(lambda z, t: z, z0, 8, "euler").final
        midpoint = integrate(lambda z, t: z, z0, 8, "midpoint").final
        self.assertLess((midpoint - exact).abs().max(), (euler - exact).abs().max())

    def test_data_reproducibility_and_shape(self):
        for dataset in ("moons", "gmm8"):
            a = sample_target(97, dataset, generator(11))
            b = sample_target(97, dataset, generator(11))
            self.assertTrue(torch.equal(a, b)); self.assertEqual(a.shape, (97, 2))
            self.assertEqual(a.dtype, torch.float32)
            self.assertFalse(torch.equal(a, sample_target(97, dataset, generator(12))))
        self.assertTrue(torch.equal(sample_base(31, generator(3)), sample_base(31, generator(3))))

    def test_model_and_loss_shapes(self):
        model = VelocityMLP(16, 2)
        z, t = torch.zeros(9, 2), torch.zeros(9, 1)
        output = model(z, t)
        self.assertEqual(output.shape, (9, 2))
        velocity_mse(output, torch.ones_like(output)).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(z, torch.zeros(9))

    def test_metric_identity(self):
        points = sample_base(32, generator(1))
        directions = torch.eye(2)
        self.assertEqual(sliced_wasserstein_2(points, points, directions), 0.)
        self.assertGreater(sliced_wasserstein_2(points + 1, points, directions), 0.)

    def test_config_and_solver_reject_invalid_input(self):
        with self.assertRaises(ValueError): TrainConfig.from_dict({"step": 5})
        with self.assertRaises(ValueError): TrainConfig(steps=0)
        with self.assertRaises(ValueError): integrate(lambda z,t:z, torch.zeros(1,2), 0)
        with self.assertRaises(ValueError): integrate(lambda z,t:z, torch.zeros(1,2), 2, "rk4")

    def test_metric_rejects_invalid_and_normalizes_directions(self):
        points = torch.zeros(4, 2)
        shifted = points + 100
        for directions in (torch.zeros(3, 2), torch.empty(0, 2), torch.full((3, 2), float("nan"))):
            with self.assertRaises(ValueError):
                sliced_wasserstein_2(points, shifted, directions)
        for bad in (torch.empty(0, 2), torch.full((4, 2), float("inf"))):
            with self.assertRaises(ValueError):
                sliced_wasserstein_2(bad, bad, torch.eye(2))
        self.assertEqual(sliced_wasserstein_2(points, shifted, 100 * torch.eye(2)), 100.)

    def test_config_rejects_wrong_types_and_nonfinite_values(self):
        for value in ([], {}, None):
            with self.assertRaises(ValueError): TrainConfig.from_dict({"dataset": value})
        for value in (None, [], "moons"):
            with self.assertRaises(ValueError): TrainConfig.from_dict(value)
        for name in ("noise", "learning_rate"):
            for value in (None, "0.1", True, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    TrainConfig.from_dict({name: value})
