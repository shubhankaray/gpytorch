import torch
from torch.autograd import Function
from ..kernels import Distance


@torch.jit.script
def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


@torch.jit.script
def default_postprocess_script(x):
    return x


class RBFKernel(Function):
    def __init__(self, lengthscale):
        self.lengthscale = lengthscale
        self.distance_module = Distance()

    def _covar_dist(
        self,
        x1,
        x2,
        diag=False,
        batch_dims=None,
        square_dist=False,
        dist_postprocess_func=default_postprocess_script,
        **params
    ):
        """
        This is a helper method for computing the Euclidean distance between
        all pairs of points in x1 and x2.
        The dimensionality of the output depends on the
        Args:
            - :attr:`x1` (Tensor `n x d` or `b x n x d`)
            - :attr:`x2` (Tensor `m x d` or `b x m x d`) - for diag mode, these must be the same inputs
            - :attr:`diag` (bool):
                Should we return the whole distance matrix, or just the diagonal?
            - :attr:`batch_dims` (tuple, optional):
                If this option is passed in, it will tell the tensor which of the
                three dimensions are batch dimensions.
                Currently accepts: standard mode (either None or (0,))
                or (0, 2) for use with Additive/Multiplicative kernels
            - :attr:`square_dist` (bool):
                Should we square the distance matrix before returning?
        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False` and `batch_dims=None`: (`b x n x n`)
            * `diag=False` and `batch_dims=(0, 2)`: (`bd x n x n`)
            * `diag=True` and `batch_dims=None`: (`b x n`)
            * `diag=True` and `batch_dims=(0, 2)`: (`bd x n`)
        """
        if batch_dims == (0, 2):
            x1 = x1.unsqueeze(0).transpose(0, -1)
            x2 = x2.unsqueeze(0).transpose(0, -1)

        x1_eq_x2 = torch.equal(x1, x2)

        res = None

        # Cache the Distance object or else JIT will recompile every time
        if not self.distance_module:
            self.distance_module = Distance(dist_postprocess_func)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                # We are computing distances between points and themselves, which are all zero
                if batch_dims == (0, 2):
                    res = torch.zeros(x1.shape[0] * x1.shape[-1], x2.shape[-2], dtype=x1.dtype, device=x1.device)
                else:
                    res = torch.zeros(x1.shape[0], x2.shape[-2], dtype=x1.dtype, device=x1.device)
                return dist_postprocess_func(res)
            else:
                if square_dist:
                    res = (x1 - x2).pow(2).sum(-1)
                else:
                    res = torch.norm(x1 - x2, p=2, dim=-1)

            res = dist_postprocess_func(res)
        elif not square_dist:
            res = self.distance_module._jit_dist(x1, x2, torch.tensor(diag), torch.tensor(x1_eq_x2))
        else:
            res = self.distance_module._jit_sq_dist(x1, x2, torch.tensor(diag), torch.tensor(x1_eq_x2))

        if batch_dims == (0, 2):
            if diag:
                res = res.transpose(0, 1).contiguous().view(-1, res.shape[-1])
            else:
                res = res.transpose(0, 1).contiguous().view(-1, *res.shape[-2:])

        return res

    def forward(self, x1, x2, lengthscale, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        # save for backward
        self.x1_ = x1_
        self.x2_ = x2_
        diff = self._covar_dist(x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, **params)
        return diff

    def backward(self, x):
        scaled_dist_sq = -2 torch.log(x)
        # TODO:
