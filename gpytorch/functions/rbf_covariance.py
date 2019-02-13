import torch


class RBFCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, lengthscale, sq_dist_func):
        if lengthscale.size(-1) > 1:
            raise ValueError("RBFCovariance cannot handle multiple lengthscales")
        x1_ = x1.div(lengthscale)
        x2_ = x2.div(lengthscale)
        unitless_sq_dist = sq_dist_func(x1_, x2_)
        # clone because inplace operations will mess with what's saved for backward
        covar_mat = unitless_sq_dist.clone().div_(-2.).exp_()
        ctx.save_for_backward(covar_mat, unitless_sq_dist, lengthscale)
        ctx.sq_dist_func = sq_dist_func
        return covar_mat

    @staticmethod
    def backward(ctx, grad_output):
        if any(ctx.needs_input_grad[:2]):
            raise RuntimeError("RBFCovariance cannot compute gradients with "
                               "respect to x1 and x2")
        covar_mat, unitless_sq_dist, lengthscale = ctx.saved_tensors
        lengthscale_grad = grad_output * covar_mat * unitless_sq_dist.div(lengthscale)
        return None, None, lengthscale_grad, None
