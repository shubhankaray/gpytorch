#include "_gpytorch_solver_base.h"
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")


// Actual function definitions
std::tuple<at::Tensor, at::Tensor> batch_symeig_cuda(at::Tensor mat);
std::tuple<at::Tensor, at::Tensor> _batch_flattened_symeig_cuda_kernel(
  at::Tensor & evals_out, at::Tensor & evecs_out
);
