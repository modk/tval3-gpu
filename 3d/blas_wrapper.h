/*
 * blas_wrapper.h
 *
 */

cublasStatus_t status;

//-----------------------------------------------------------------------------
// cublas_scal
//-----------------------------------------------------------------------------
inline cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                                  const float *alpha, float *x, int incx) {
#ifdef PROFILING
	profile_info[40].valid = true;
	sprintf(profile_info[40].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status = cublasSscal(handle, n, alpha, x, incx);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[40].time += elapsedTime;
	profile_info[40].runs++;
#endif
	return status;
}

inline cublasStatus_t cublas_scal(cublasHandle_t handle, int n,
                                  const double *alpha, double *x, int incx) {
#ifdef PROFILING
	profile_info[40].valid = true;
	sprintf(profile_info[40].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status = cublasDscal(handle, n, alpha, x, incx);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[40].time += elapsedTime;
	profile_info[40].runs++;
#endif
	return status;
}
//-----------------------------------------------------------------------------
// cublas_axpy
//-----------------------------------------------------------------------------
inline cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                                  const float *alpha, const float *x, int incx,
                                  float *y, int incy) {
#ifdef PROFILING
	profile_info[41].valid = true;
	sprintf(profile_info[41].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status = cublasSaxpy(handle, n, alpha, x, incx, y, incy );
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[41].time += elapsedTime;
	profile_info[41].runs++;
#endif
	return status;
}

inline cublasStatus_t cublas_axpy(cublasHandle_t handle, int n,
                                  const double *alpha, const double *x,
                                  int incx, double *y, int incy) {
#ifdef PROFILING
	profile_info[41].valid = true;
	sprintf(profile_info[41].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status = cublasDaxpy(handle, n, alpha, x, incx, y, incy );
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[41].time += elapsedTime;
	profile_info[41].runs++;
#endif
	return status;
}

//-----------------------------------------------------------------------------
// cublas_dot
//-----------------------------------------------------------------------------
inline cublasStatus_t cublas_dot(cublasHandle_t handle, int n, const float *x,
                                 int incx, const float *y, int incy,
                                 float *result) {
#ifdef PROFILING
	profile_info[42].valid = true;
	sprintf(profile_info[42].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status = cublasSdot(handle, n, x, incx, y, incy, result);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[42].time += elapsedTime;
	profile_info[42].runs++;
#endif
	return status;
}

inline cublasStatus_t cublas_dot(cublasHandle_t handle, int n, const double *x,
                                 int incx, const double *y, int incy,
                                 double *result) {
#ifdef PROFILING
	profile_info[42].valid = true;
	sprintf(profile_info[42].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status = cublasDdot(handle, n, x, incx, y, incy, result);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[42].time += elapsedTime;
	profile_info[42].runs++;
#endif
	return status;
}

//-----------------------------------------------------------------------------
// cublas_nrm2
//-----------------------------------------------------------------------------
inline cublasStatus_t cublas_nrm2(cublasHandle_t handle, int n, const float *x,
                                  int incx, float *result ) {
#ifdef PROFILING
	profile_info[43].valid = true;
	sprintf(profile_info[43].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status = cublasSnrm2(handle, n, x, incx, result );
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[43].time += elapsedTime;
	profile_info[43].runs++;
#endif
	return status;
}

inline cublasStatus_t cublas_nrm2(cublasHandle_t handle, int n, const double *x,
                                  int incx, double *result ) {
#ifdef PROFILING
	profile_info[43].valid = true;
	sprintf(profile_info[43].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status = cublasDnrm2(handle, n, x, incx, result );
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[43].time += elapsedTime;
	profile_info[43].runs++;
#endif
	return status;
}

//-----------------------------------------------------------------------------
// cublas_gemv
//-----------------------------------------------------------------------------
inline cublasStatus_t cublas_gemv(cublasHandle_t handle,
                                  cublasOperation_t trans, int m, int n,
                                  const float *alpha,
                                  const float *A, int lda, const float *x,
                                  int incx, const float *beta, float *y,
                                  int incy) {
#ifdef PROFILING
	profile_info[44].valid = true;
	sprintf(profile_info[44].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status = cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta,
	                     y, incy);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[44].time += elapsedTime;
	profile_info[44].runs++;
#endif
	return status;
}

inline cublasStatus_t cublas_gemv(cublasHandle_t handle,
                                  cublasOperation_t trans, int m, int n,
                                  const double *alpha,
                                  const double *A, int lda, const double *x,
                                  int incx, const double *beta, double *y,
                                  int incy) {
#ifdef PROFILING
	profile_info[44].valid = true;
	sprintf(profile_info[44].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status = cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta,
	                     y, incy);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[44].time += elapsedTime;
	profile_info[44].runs++;
#endif
	return status;
}

cusparseStatus_t status0;

//-----------------------------------------------------------------------------
// cusparse_csrmv
//-----------------------------------------------------------------------------
inline cusparseStatus_t cusparse_csrmv(cusparseHandle_t handle,
                                       cusparseOperation_t transA, int m, int n,
                                       int nnz,
                                       const float *alpa,
                                       const cusparseMatDescr_t descrA,
                                       const float *val, const int *ptr,
                                       const int *ind, const float *x,
                                       const float *beta, float *y) {
#ifdef PROFILING0
	profile_info[45].valid = true;
	sprintf(profile_info[45].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status0 = cusparseScsrmv( handle, transA, m, n, nnz, alpa, descrA, val,
	                          ptr, ind, x, beta, y);
#ifdef PROFILING0
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[45].time += elapsedTime;
	profile_info[45].runs++;
#endif
	return status0;
}

inline cusparseStatus_t cusparse_csrmv(cusparseHandle_t handle,
                                       cusparseOperation_t transA, int m, int n,
                                       int nnz,
                                       const double *alpa,
                                       const cusparseMatDescr_t descrA,
                                       const double *val, const int *ptr,
                                       const int *ind, const double *x,
                                       const double *beta, double *y) {
#ifdef PROFILING0
	profile_info[45].valid = true;
	sprintf(profile_info[45].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status0 = cusparseDcsrmv( handle, transA, m, n, nnz, alpa, descrA, val,
	                          ptr, ind, x, beta, y);
#ifdef PROFILING0
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[45].time += elapsedTime;
	profile_info[45].runs++;
#endif
	return status0;
}

//-----------------------------------------------------------------------------
// cusparseShybmv
//-----------------------------------------------------------------------------
inline cusparseStatus_t cusparse_hybmv(cusparseHandle_t handle,
                                       cusparseOperation_t transA,
                                       const float *alpa,
                                       const cusparseMatDescr_t descrA,
                                       const cusparseHybMat_t hybA,
                                       const float *x,
                                       const float *beta, float *y) {
#ifdef PROFILING
	profile_info[46].valid = true;
	sprintf(profile_info[46].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status0 =
	        cusparseShybmv(handle, transA, alpa, descrA, hybA, x, beta, y);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[46].time += elapsedTime;
	profile_info[46].runs++;
#endif
	return status0;
}

inline cusparseStatus_t cusparse_hybmv(cusparseHandle_t handle,
                                       cusparseOperation_t transA,
                                       const double *alpa,
                                       const cusparseMatDescr_t descrA,
                                       const cusparseHybMat_t hybA,
                                       const double *x,
                                       const double *beta, double *y) {
#ifdef PROFILING
	profile_info[46].valid = true;
	sprintf(profile_info[46].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start));
#endif
	status0 =
	        cusparseDhybmv(handle, transA, alpa, descrA, hybA, x, beta, y);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	profile_info[46].time += elapsedTime;
	profile_info[46].runs++;
#endif
	return status0;
}

