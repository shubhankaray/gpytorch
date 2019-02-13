#include "gpytorch_solver_cuda.h"


std::tuple<at::Tensor, at::Tensor> batch_symeig_cuda(at::Tensor mat) {
  CHECK_CUDA(mat);
  CHECK_CONTIGUOUS(mat);
  CHECK_SQUARE(mat);

  auto batch_size = batchCount(mat);
  at::Tensor && evals = at::empty({batch_size, mat.size(-2)}, mat.options());
  at::Tensor && evecs = at::empty({batch_size, mat.size(-2), mat.size(-1)}, mat.options());
  at::Tensor && flattened_mat = mat.reshape({batch_size, mat.size(-2), mat.size(-1)});

	evecs.copy_(flattened_mat);
  _batch_flattened_symeig_cuda_kernel(evals, evecs);

  at::IntList evecs_shape = mat.sizes();
  at::IntList evals_shape = at::IntList(evecs_shape.data(), evecs_shape.size() - 1);
  evecs.resize_(evecs_shape);
  evals.resize_(evals_shape);
  return std::make_tuple(evals, evecs);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_symeig_cuda", &batch_symeig_cuda, "Batch symeig solver (CUDA)");
}
