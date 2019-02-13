#include "gpytorch_solver_cuda.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>


// Parameters
const double tol = 1.e-7;
const int max_sweeps = 15;
const int sort_eig = 1;
const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;


std::tuple<at::Tensor, at::Tensor> _batch_flattened_symeig_cuda_kernel(
  at::Tensor & evals_out, at::Tensor & mat,
) {
	cusolverDnHandle_t cusolver_h = NULL;
	cudaStream_t stream = NULL;
	syevjInfo_t syevj_params = NULL;

	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cuda_stat = cudaSuccess;

	const int matrix_size = mat.size(1);  // == mat.size(2)
	const int batch_size = mat.size(0);

	/* step 1: create cusolver handle, bind a stream  */
	status = cusolverDnCreate(&cusolver_h);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cuda_stat = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == cuda_stat);

	status = cusolverDnSetStream(cusolver_h, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: configuration of syevj */
	status = cusolverDnCreateSyevjInfo(&syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of tolerance is machine zero */
	status = cusolverDnXsyevjSetTolerance(syevj_params, tol);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of max. sweeps is 100 */
	status = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* sorting */
	status = cusolverDnXsyevjSetSortEig(syevj_params, sort_eig);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	// Get data
  auto evecs_data = evecs_out.data()
  auto evals_data = evals_out.data()

	/* step 4: query working space of syevjBatched */
  int lwork = 0;
  double *work = NULL;

	status = cusolverDnDsyevjBatched_bufferSize(
			cusolver_h,
			jobz,
			uplo,
			matrix_size,
			evec_data,
			matrix_size,
			evals_data,
			&lwork,
			syevj_params,
			batch_size
	);
	assert(CUSOLVER_STATUS_SUCCESS == status);

  cuda_stat = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
  assert(cudaSuccess == cuda_stat);

  return std::make_tuple(evals_out, evecs_out);
}
