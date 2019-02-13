import torch
import math


class MaternCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, lengthscale, nu, dist_func):
        # Subtract mean for numerical stability. Won't affect computations
        # because covariance matrix is stationary
        mean = x1.contiguous().view(-1, 1, x1.size(-1)).mean(0, keepdim=True)
        x1_ = (x1 - mean).div(lengthscale)
        x2_ = (x2 - mean).div(lengthscale)
        scaled_unitless_dist = dist_func(x1_, x2_).mul_(math.sqrt(2 * nu))
        # clone because inplace operations will mess with what's saved for backward
        exp_component = scaled_unitless_dist.clone().mul_(-1).exp_()

        if nu == 0.5:
            covar_mat = exp_component
        elif nu == 1.5:
            covar_mat = exp_component.mul_(1 + scaled_unitless_dist)
        elif nu == 2.5:
            covar_mat = exp_component.mul_(1 + scaled_unitless_dist
                                           + scaled_unitless_dist.pow(2).div(3))
        nu = torch.tensor(nu)
        ctx.save_for_backward(scaled_unitless_dist, lengthscale, nu)
        return covar_mat

    @staticmethod
    def backward(ctx, grad_output):
        if any(ctx.needs_input_grad[:2]):
            raise RuntimeError("MaternCovariance cannot compute gradients with "
                               "respect to x1 and x2")
        scaled_unitless_dist, lengthscale, nu = ctx.saved_tensors
        exp_component = torch.exp(-scaled_unitless_dist)
        if nu == 0.5:
            lengthscale_grad = grad_output * exp_component * scaled_unitless_dist.div(lengthscale)
        elif nu == 1.5:
            lengthscale_grad = grad_output * exp_component * scaled_unitless_dist.pow(2).div(lengthscale)
        elif nu == 2.5:
            lengthscale_grad = (
                grad_output * exp_component * scaled_unitless_dist.pow(2).div(3 * lengthscale)
                * (1 + scaled_unitless_dist)
            )
        return None, None, lengthscale_grad, None, None
