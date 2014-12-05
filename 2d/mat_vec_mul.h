/*
 * mat_vec_mul.h
 *
 *  Created on: Jul 13, 2011
 *      Author: ditlevsen
 */

#ifndef MAT_VEC_MUL_H_
#define MAT_VEC_MUL_H_

struct sparse_mm {
	sparse_mat_device A_csc;
	sparse_mat_device A_csr;
	int layers;
	cusparseHandle_t cs_handle;
	cusparseMatDescr_t descrA;
	sparse_mm(const sparse_mat_host &A, int l, cusparseHandle_t handle,
	          cusparseMatDescr_t descriptor) :
		A_csc(A),
		A_csr(A.rows, A.cols, A.nnz, sparse_mat_csr,
		      false), layers(l), cs_handle(handle),
		descrA(descriptor) {
		HANDLE_ERROR(cusparseScsr2csc(cs_handle, A.cols, A.rows,
		                              A_csc.val(),
		                              A_csc.ptr(), A_csc.ind(),
		                              A_csr.val(), A_csr.ind(),
		                              A_csr.ptr(), 1,
		                              CUSPARSE_INDEX_BASE_ZERO));
	};
};

struct dense_mm {
	mat_device A;
	int layers;
	cublasHandle_t cb_handle;
	dense_mm(const mat_host &A_host, int l, cublasHandle_t handle) : A(
		        A_host, true), layers(l), cb_handle(handle) {
	};
};

inline cusparseStatus_t mat_vec_mul(cublasOperation_t transA,
                                    const sparse_mm &A, const float *x,
                                    float *y,
                                    cudaStream_t stream = 0) {
	int n, m;
	const sparse_mat_device *mA;

	if (transA == CUBLAS_OP_N) {
		n = A.A_csr.cols;
		m = A.A_csr.rows;
		mA = &A.A_csr;
	} else {
		n = A.A_csr.rows;
		m = A.A_csr.cols;
		mA = &A.A_csc; // implicit tranponation...
	}

	HANDLE_ERROR(cusparseSetKernelStream(A.cs_handle, stream));

	if (A.layers == 1)
		return cusparseScsrmv(A.cs_handle,
		                      CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, 1,
		                      A.descrA, mA->val(),
		                      mA->ptr(), mA->ind(), x, 0, y);
	else {
		return cusparseScsrmm(A.cs_handle,
		                      CUSPARSE_OPERATION_NON_TRANSPOSE, m,
		                      A.layers, n, 1, A.descrA,
		                      mA->val(), mA->ptr(),
		                      mA->ind(), x, n, 0, y, m);

	}
}

inline cublasStatus_t mat_vec_mul(cublasOperation_t trans, const dense_mm &A,
                                  const float *x, float *y,
                                  cudaStream_t stream = 0) {
	int n = A.A.cols;
	int m = A.A.rows;
	cublasStatus_t status;

	// cublasSgemv: m = number of columns of A
	// cublasSgemm: m = number of columns of op(A)...
	int N = (trans == CUBLAS_OP_N) ? n : m;
	int M = (trans == CUBLAS_OP_N) ? m : n;

	float alpha = 1, beta = 0;

	HANDLE_ERROR(cublasSetStream(A.cb_handle, stream));

	if (A.layers == 1)
		status = cublasSgemv(A.cb_handle, trans, m, n, &alpha,
		                     A.A.data_dev_ptr(),
		                     A.A.leading_dim(), x, 1, &beta, y,
		                     1);
	else
		status = cublasSgemm(A.cb_handle, trans, CUBLAS_OP_N, M,
		                     A.layers, N, &alpha, A.A.data_dev_ptr(),
		                     A.A.leading_dim(), x, N, &beta, y, M);

	HANDLE_ERROR(cublasSetStream(A.cb_handle, 0));

	return status;
}


#endif /* MAT_VEC_MUL_H_ */
