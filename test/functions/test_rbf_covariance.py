#!/usr/bin/env python3

import torch
import unittest
import gpytorch


def sq_dist_func(x1, x2):
    dist_module = gpytorch.kernels.kernel.Distance()
    return dist_module._jit_sq_dist(x1, x2, torch.tensor(torch.equal(x1, x2)))


class TestRBFCovariance(unittest.TestCase):
    def test_forward(self):
        x1 = torch.randn(2, 5, 3)
        x2 = torch.randn(2, 6, 3)
        # Doesn't support ARD
        lengthscale = torch.tensor([1.5, 3.2]).view(2, 1, 1)

        res = gpytorch.functions.RBFCovariance().apply(x1, x2, lengthscale, sq_dist_func)
        actual = sq_dist_func(x1, x2).div(-2 * lengthscale**2).exp()
        self.assertTrue(torch.allclose(res, actual))

    def test_backward(self):
        x1 = torch.randn(2, 5, 3, dtype=torch.float64)
        x2 = torch.randn(2, 6, 3, dtype=torch.float64)
        lengthscale = torch.randn(2, dtype=torch.float64, requires_grad=True).view(2, 1, 1)
        f = lambda x1, x2, l: gpytorch.functions.RBFCovariance().apply(x1, x2, l, sq_dist_func)
        try:
            torch.autograd.gradcheck(f, (x1, x2, lengthscale))
        except RuntimeError:
            self.fail("Gradcheck failed")


if __name__ == "__main__":
    unittest.main()
