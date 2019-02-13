#!/usr/bin/env python3

import torch
from _gpytorch_solver import batch_symeig


def batch_symeig(mat):
    """
    """
    with torch.no_grad():
        mat_orig = mat
        batch_shape = torch.Size(mat_orig.shape[:-2])
        matrix_shape = torch.Size(mat_orig.shape[-2:])

        # Smaller matrices are faster on the CPU than the GPU
        # if mat.size(-1) <= 32:
            # mat = mat.cpu()

        mat = mat.view(-1, *matrix_shape)
        eigenvectors = torch.empty(batch_shape.numel(), *matrix_shape, dtype=mat.dtype, device=mat.device)
        eigenvalues = torch.empty(batch_shape.numel(), matrix_shape[-1], dtype=mat.dtype, device=mat.device)

        for i in range(batch_shape.numel()):
            evals, evecs = mat[i].symeig(eigenvectors=True)
            # mask = evals.ge(0)
            # eigenvectors[i] = evecs * mask.type_as(evecs).unsqueeze(0)
            # eigenvalues[i] = evals.masked_fill_(1 - mask, 1)
            eigenvectors[i] = evecs
            eigenvalues[i] = evals

        return eigenvalues.view(*batch_shape, -1), eigenvectors.view_as(mat_orig)
