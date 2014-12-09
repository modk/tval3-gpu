#ifndef MAT_VEC_MUL_DENSE_H_
#define MAT_VEC_MUL_DENSE_H_

/******************************************************************************
* Single-GPU processing for dense matrices (makes not much sense due to 
* extensive matrix size, but nevertheless working...)
*******************************************************************************/

#if (1)
//-----------------------------------------------------------------------------
// type definition for dense matrices on single GPU
//-----------------------------------------------------------------------------
template<class Type>
struct dense_mm {
	mat_device<Type> A;
	cublasHandle_t cb_handle;
	dense_mm(const mat_host<Type> &A_host, cublasHandle_t handle) : A(
		        A_host, true), cb_handle(handle) {
	};
};

//-----------------------------------------------------------------------------
// mat_vec_mul for dense matrices
//-----------------------------------------------------------------------------
template<class Type>
inline cublasStatus_t mat_vec_mul(cublasOperation_t trans,
                                  const dense_mm<Type> &A, const Type *x,
                                  Type *y, cudaStream_t stream = 0,
                                  bool mulChain = false) {
#ifdef PROFILING
	if (trans == CUBLAS_OP_N) {
		profile_info[4].valid = true;
		sprintf(profile_info[4].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 4 );
	} else {
		profile_info[5].valid = true;
		sprintf(profile_info[5].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 7 );
	}
	HANDLE_ERROR(cudaEventRecord(start));
#endif

	int n = A.A.dim_x;
	int m = A.A.dim_y;
	cublasStatus_t status;

	Type alpha = 1, beta = 0;

	HANDLE_ERROR(cublasSetStream(A.cb_handle, stream));

	status = cublas_gemv(A.cb_handle, trans, m, n, &alpha,
	                     A.A.data_dev_ptr(),
	                     A.A.leading_dim(), x, 1, &beta, y, 1);

	HANDLE_ERROR(cublasSetStream(A.cb_handle, 0));

#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	if (trans == CUBLAS_OP_N) {
		profile_info[4].time += elapsedTime;
		profile_info[4].runs++;
	} else {
		profile_info[5].time += elapsedTime;
		profile_info[5].runs++;
	}
#endif

	return status;
}
#endif


#endif /* MAT_VEC_MUL_DENSE_H_ */
