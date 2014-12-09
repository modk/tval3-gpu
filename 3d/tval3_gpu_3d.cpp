#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "handle_error.h"
#include "container_device.h"
#include "tval3_gpu_3d.h"
#include <fstream>

#define PROFILING
#define _PROFILE_VARS_
#include "profile_gpu.h"
//#include "tval3_gpu_3d_kernels.cuh"
#include "mat_vec_mul.h"
#include "blas_wrapper.h"

using namespace std;

#ifdef PROFILING
float t_mult_t, t_mult_nt, t_mult_gettau, t_d, t_dt; bool rec_time;
cudaEvent_t start_part, stop_part;
#endif

extern const tval3_info<float> tval3_gpu_3d(mat_host<float> &U,
		const sparse_mat_host<float> &A, const mat_host<float> &b,
		const tval3_options<float> &opts, const mat_host<float> &Ut,
		bool pagelocked, int mainGPU, int numGPUs,
		int *gpuArray ) __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_gpu_3d(mat_host<float> &U,
		const mat_host<float> &A, const mat_host<float> &b,
		const tval3_options<float> &opts, const mat_host<float> &Ut,
		bool pagelocked, int mainGPU, int numGPUs, int *gpuArray ) 
	__attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_gpu_3d(mat_host<float> &U,
		const geometry_host &A, const mat_host<float> &b,
		const tval3_options<float> &opts, const mat_host<float> &Ut,
		bool pagelocked, int mainGPU, int numGPUs,
		int *gpuArray ) __attribute__ ((visibility ("default") ));

extern const tval3_info<float> tval3_gpu_3d(mat_host<float> &U,
		const geometry_host &A_geo, sparse_mat_host<float> &A_mat,
		const mat_host<float> &b, const tval3_options<float> &opts,
		const mat_host<float> &Ut, bool pagelocked, int mainGPU,
		int numGPUs, int *gpuArray )  __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_gpu_3d(mat_host<double> &U,
		const sparse_mat_host<double> &A, const mat_host<double> &b,
		const tval3_options<double> &opts, const mat_host<double> &Ut,
		bool pagelocked, int mainGPU, int numGPUs, int *gpuArray ) 
	__attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_gpu_3d(mat_host<double> &U,
		const mat_host<double> &A, const mat_host<double> &b,
		const tval3_options<double> &opts, const mat_host<double> &Ut,
		bool pagelocked, int mainGPU, int numGPUs, int *gpuArray ) 
	__attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_gpu_3d(mat_host<double> &U,
		const geometry_host &A, const mat_host<double> &b,
		const tval3_options<double> &opts, const mat_host<double> &Ut,
		bool pagelocked, int mainGPU, int numGPUs,
		int *gpuArray ) __attribute__ ((visibility ("default") ));

extern const tval3_info<double> tval3_gpu_3d(mat_host<double> &U,
		const geometry_host &A_geo, sparse_mat_host<double> &A_mat,
		const mat_host<double> &b, const tval3_options<double> &opts,
		const mat_host<double> &Ut, bool pagelocked, int mainGPU,
		int numGPUs, int *gpuArray ) __attribute__ ((visibility ("default") ));

cudaStream_t stream1, stream2, stream3;

int majorRevision;

cublasHandle_t cb_handle;
cusparseHandle_t cs_handle;
cusparseMatDescr_t descrA;
cusparseHybMat_t hybA;
cusparseHybMat_t hybB;

// this struct holds various values, accessed by the functions get_g(), 
// update_g(), update_W() and update_mlp()
// the member names correspond to the variable names in the MATLAB version

template<class Type>
struct tval3_data_gpu { // parameters and intermediate results
	// dimensions
	int p; // y-dimension of reconstruction volume
	int q; // x-dimension of reconstruction volume
	int r; // z-dimension of reconstruction volume
	int n; // number of pixels (n = p*q*r)
	int m; // number of measurements

	// penalty parameters
	Type mu;
	Type beta;
	Type muDbeta; // mu/beta

	// multiplier
	mat_device<Type> delta;
	mat_device<Type> sigmaX;
	mat_device<Type> sigmaY;
	mat_device<Type> sigmaZ;

	// lagrangian function and sub terms
	Type f;
	Type lam1;
	Type lam2;
	Type lam3;
	Type lam4;
	Type lam5;

	// other intermediate results
	mat_device<Type> Up;
	mat_device<Type> dU; // U-Up, after steepest descent
	mat_device<Type> uup; // U-Up, after "backtracking"
	mat_device<Type> Ux; // gradients in x-direction
	mat_device<Type> Uxp;
	mat_device<Type> dUx;
	mat_device<Type> Uy; // gradients in y-direction
	mat_device<Type> Uyp;
	mat_device<Type> dUy;
	mat_device<Type> Uz; // gradients in z-direction
	mat_device<Type> Uzp;
	mat_device<Type> dUz;
	mat_device<Type> Wx;
	mat_device<Type> Wy;
	mat_device<Type> Wz;
	mat_device<Type> Atb; // A'*b
	mat_device<Type> Au; // A*u
	mat_device<Type> Aup;
	mat_device<Type> dAu;
	mat_device<Type> d; // gradient of objective function (2.28)
	mat_device<Type> g; // sub term of (2.28)
	mat_device<Type> gp;
	mat_device<Type> dg;
	mat_device<Type> g2; // sub term of (2.28)
	mat_device<Type> g2p;
	mat_device<Type> dg2;

	// buffers for reductions
	mat_device<Type> gpu_buff1; // length: |U|
	mat_device<Type> gpu_buff2; // length: |U|
	mat_device<Type> gpu_buff3; // length: |b|
	mat_device<Type> gpu_buff4; // length: |b|
	mat_host<Type> host_buff1; // length: |U|
	mat_host<Type> host_buff2; // length: |U|
	mat_host<Type> host_buff3; // length: |b|
	mat_host<Type> host_buff4; // length: |b|

	// additional parameters for skipping of multiplication with adjunct matrix
	const unsigned int numInitIterations;
	const Type skipMulRatio;
	const Type maxRelativeChange;
	unsigned int currInitIteration;
	Type currSkipValue;
	mat_device<Type> last_Au;

	tval3_data_gpu(const mat_host<Type> &U, const mat_host<Type> &b,
	               const tval3_options<Type> &opts, bool pagelocked) :
		p(U.dim_y), q(U.dim_x), r(U.dim_z), n(U.len), m(b.len), 
		mu(opts.mu0), beta(opts.beta0), delta(b.len), 
		sigmaX( U.dim_y, U.dim_x, U.dim_z), 
		sigmaY(U.dim_y, U.dim_x, U.dim_z), 
		sigmaZ(U.dim_y, U.dim_x, U.dim_z), 
		Up(U.dim_y, U.dim_x, U.dim_z, false),

		dU(U.dim_y, U.dim_x, U.dim_z, false), 
		uup(U.len), 
		Ux(U.dim_y, U.dim_x, U.dim_z, false),
		Uxp(U.dim_y, U.dim_x, U.dim_z, false),
		dUx(U.dim_y, U.dim_x, U.dim_z, false), 
		Uy(U.dim_y, U.dim_x, U.dim_z, false), 
		Uyp(U.dim_y, U.dim_x, U.dim_z, false),
		dUy(U.dim_y, U.dim_x, U.dim_z, false), 
		Uz(U.dim_y, U.dim_x, U.dim_z, false), 
		Uzp(U.dim_y, U.dim_x, U.dim_z, false),
		dUz(U.dim_y, U.dim_x, U.dim_z, false), 
		Wx(U.dim_y, U.dim_x, U.dim_z, false), 
		Wy(U.dim_y, U.dim_x, U.dim_z, false),
		Wz(U.dim_y, U.dim_x, U.dim_z, false), 
		Atb(U.len, false), Au(b.len, false), 
		Aup(b.len, false), dAu(b.len, false),
		d(U.dim_y, U.dim_x, U.dim_z, false, false), 
		g(U.len, false), gp(U.len, false), 
		dg(U.len, false), g2(U.len, false), g2p(U.len, false),
		dg2(U.len, false), gpu_buff1(U.len, false), gpu_buff2(U.len,
		                                                      false),
		host_buff1(U.len, false, pagelocked),
		host_buff2(U.len, false, pagelocked),
		gpu_buff3(b.len, false), gpu_buff4(b.len, false), host_buff3(
		        b.len, false, pagelocked),
		host_buff4(b.len, false, pagelocked),
		numInitIterations(opts.numInitIterations), skipMulRatio(
		        opts.skipMulRatio), maxRelativeChange(
		        opts.maxRelativeChange),
		currInitIteration(0), currSkipValue(0.0), last_Au(b.len) {
	}
};

//-----------------------------------------------------------------------------
// vec_add_gpu: z <- alpha*x + y, N: number of elements
//-----------------------------------------------------------------------------
template<class Type>
inline void vec_add_gpu(const int N, const int stride, const Type *y,
                        const Type alpha, const Type *x, Type *z,
                        cudaStream_t stream = 0) {
	int threads, blocks;

	if (majorRevision >= 2) {
		threads = 256;
		blocks = max((int)round((double)N / 256 / 2 / 16) * 16, 16);
	} else {
		threads = 128;
		blocks = 120;
	}
	blocks = min(blocks, (N + threads - 1) / threads);

	vec_add_kernel_wrapper(blocks, threads, 0, stream, N, z, alpha, x, y, NULL);
}

//-----------------------------------------------------------------------------
// scaleb_gpu: Scales the measurement vector. Returns the scale factor.
//-----------------------------------------------------------------------------
template<class Type>
Type scaleb_gpu(const mat_host<Type> &b, mat_device<Type> &b_gpu) {
	Type scl = 1;

	if (b.len > 0) {
		Type threshold1 = 0.5;
		Type threshold2 = 1.5;
		scl = 1;
		Type val;
		Type val_abs;
		Type bmin = b[0];
		Type bmax = bmin;
		Type bmin_abs = abs(bmin);
		Type bmax_abs = bmin_abs;
		for (int i = 0; i < b.len; i++) {
			val = b[i];
			val_abs = abs(val);

			if (val_abs < bmin_abs) {
				bmin = val;
				bmin_abs = val_abs;
			}

			if (val_abs > bmax_abs) {
				bmax = val;
				bmax_abs = val_abs;
			}
		}
		Type b_dif = abs(bmax - bmin);

		if (b_dif < threshold1) {
			scl = threshold1 / b_dif;
			HANDLE_ERROR(cublas_scal(cb_handle, b_gpu.len, &scl,
			                         b_gpu.data_dev_ptr(), 1));
		} else if (b_dif > threshold2) {
			scl = threshold2 / b_dif;
			HANDLE_ERROR(cublas_scal(cb_handle, b_gpu.len, &scl,
			                         b_gpu.data_dev_ptr(), 1));
		}
	}
	return scl;
}

//-----------------------------------------------------------------------------
// D_gpu: matrices stored in column major format
//-----------------------------------------------------------------------------
template<class Type>
void D_gpu(mat_device<Type> &Ux, mat_device<Type> &Uy, mat_device<Type> &Uz,
           const mat_device<Type> &U) {
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(start_part));
#endif
	dim3 threads, blocks;

	if (majorRevision >= 2) {

		threads.x = 32;
		threads.y = 4;
		threads.z = 1;
		blocks.x = max((int)round((double)U.dim_x / threads.x / 6), 1);
		blocks.y = max((int)round((double)U.dim_y / threads.y), 1);
		blocks.z = max((int)round((double)U.dim_z / threads.z), 1);

		D_kernel_no_tex_wrapper( blocks, threads, 0, 0,
				U.data_dev_ptr(), Uy.data_dev_ptr(), Ux.data_dev_ptr(),
				Uz.data_dev_ptr(), U.dim_y, U.dim_x, U.dim_z);

	} else {

		threads.x = 64;
		threads.y = 4;
		threads.z = 1;
		blocks.x = max((int)round((double)U.dim_x / threads.x), 1);
		blocks.y = max((int)round((double)U.dim_y / threads.y), 1);
		blocks.z = 1;

		D_kernel_tex_wrapper(blocks, threads, 0, 0, U.len * sizeof(Type),
				U.data_dev_ptr(), Uy.data_dev_ptr(), Ux.data_dev_ptr(),
				Uz.data_dev_ptr(), U.dim_y, U.dim_x, U.dim_z);

	}
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_part));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_part));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_part, stop_part));

	if (rec_time) t_d += elapsedTime;
#endif
}

//-----------------------------------------------------------------------------
// Dt_gpu
//-----------------------------------------------------------------------------
template<class Type>
void Dt_gpu(mat_device<Type> &res, const mat_device<Type> &X,
            const mat_device<Type> &Y, const mat_device<Type> &Z,
            cudaStream_t stream = 0) {
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(start_part));
#endif
	dim3 threads, blocks;

	if (majorRevision >= 2) {
		threads.x = 32;
		threads.y = 4;
		threads.z = 1;
		blocks.x =
		        max((int)round((double)X.dim_x / threads.x * 0.4), 1);
		blocks.y =
		        max((int)round((double)X.dim_y / threads.y * 0.6), 1);
		blocks.z = max((int)round((double)X.dim_z / threads.z), 1);
		Dt_kernel_no_tex_wrapper(blocks, threads, 0, stream,
				Y.data_dev_ptr(), X.data_dev_ptr(), Z.data_dev_ptr(),
				res.data_dev_ptr(), X.dim_y, X.dim_x, X.dim_z);
	} else {
		threads.x = 32;
		threads.y = 2;
		threads.z = 8;
		blocks.x = max((int)round((double)X.dim_x / threads.x), 1);
		blocks.y =
		        max((int)round((double)X.dim_y / threads.y * 0.6), 1);
		blocks.z = 1;
		Dt_kernel_tex_wrapper(blocks, threads, 0, stream,
				X.len * sizeof(Type), Y.len * sizeof(Type), Z.len * sizeof(Type),
				Y.data_dev_ptr(), X.data_dev_ptr(), Z.data_dev_ptr(), 
				res.data_dev_ptr(), X.dim_y, X.dim_x, X.dim_z);
	}
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_part));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_part));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_part, stop_part));

	if (rec_time) t_dt += elapsedTime;
#endif
}

//-----------------------------------------------------------------------------
// get_g_gpu
//-----------------------------------------------------------------------------
template<class matrix_type, class Type>
void get_g_gpu(tval3_data_gpu<Type> &data, const matrix_type &A,
               const mat_device<Type> &U, const mat_device<Type> &b,
               bool force_mul = true) {

	// set block- and grid dimensions for lam2_4_kernel and lam3_5_kernel
	int threads = 128;
	int blocks = (majorRevision >= 2) ? 96 : 90;
	int blocks_lam2_4 = min(blocks, (data.n + threads - 1) / threads);
	int blocks_lam3_5 = min(blocks, (data.m + threads - 1) / threads);


#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(start_part));
#endif
	// Au = A*U
	HANDLE_ERROR(mat_vec_mul(CUBLAS_OP_N, A, U.data_dev_ptr(),
	                         data.Au.data_dev_ptr(), stream1, true));
	//HANDLE_ERROR(cudaStreamSynchronize(stream1));

#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_part));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_part));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_part, stop_part));

	if (rec_time) t_mult_nt += elapsedTime;
#endif

	Type relChange = 0;
	Type val_alpha = -1;

	if ( data.skipMulRatio != 0) {

		mat_device<Type> deltaAu = data.Au;

		HANDLE_ERROR(cublas_axpy(cb_handle, deltaAu.len, &val_alpha,
		                         data.last_Au.data_dev_ptr(), 1,
		                         deltaAu.data_dev_ptr(), 1));
		Type nrm_deltaAu;
		HANDLE_ERROR(cublas_nrm2(cb_handle, deltaAu.len,
		                         deltaAu.data_dev_ptr(), 1,
		                         &nrm_deltaAu));
		Type nrm_Au;
		HANDLE_ERROR(cublas_nrm2(cb_handle, data.Au.len,
		                         data.Au.data_dev_ptr(), 1, &nrm_Au));
		relChange = nrm_deltaAu / nrm_Au;

	}

#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(start_part));
#endif

	// g = A'*Au
	if ( data.skipMulRatio == 0) {

		HANDLE_ERROR(mat_vec_mul(CUBLAS_OP_T, A, data.Au.data_dev_ptr(),
		                         data.g.data_dev_ptr(), stream1, true));

	} else {

		if (force_mul == true) {
		// do always, if forced (used for initialization outside of loop and if 
		// steepest descent fails

			data.last_Au = data.Au;
			HANDLE_ERROR(mat_vec_mul(CUBLAS_OP_T, A,
						data.Au.data_dev_ptr(), data.g.data_dev_ptr(), stream1, true));
			data.currSkipValue = 0;

		}  else if (data.currInitIteration < data.numInitIterations) { 
			// do also for specified number of first iterations, after that do only, if

			data.last_Au = data.Au;
			HANDLE_ERROR(mat_vec_mul(CUBLAS_OP_T, A, data.Au.data_dev_ptr(),
						data.g.data_dev_ptr(), stream1, true));
			data.currInitIteration++;
			data.currSkipValue = 0;
			//cout << "init " << data.currInitIteration << "\n";

		} else if (data.currSkipValue < 1) {
			// specified skip-ratio is reached or if

			data.last_Au = data.Au;
			HANDLE_ERROR(mat_vec_mul(CUBLAS_OP_T, A, data.Au.data_dev_ptr(), 
						data.g.data_dev_ptr(), stream1, true));
			data.currSkipValue += data.skipMulRatio;
			//cout << "skip " << data.currSkipValue << "\n";

		} else if (relChange > data.maxRelativeChange) { // data.Au changed to much

			data.last_Au = data.Au;
			HANDLE_ERROR(mat_vec_mul(CUBLAS_OP_T, A, data.Au.data_dev_ptr(),
						data.g.data_dev_ptr(), stream1, true));
			//cout << "maxChange " << relChange << "\n";

		} else {

			data.currSkipValue -= 1;
			//cout << "..\n";
		}
	}

	//HANDLE_ERROR(cudaStreamSynchronize(stream1));

#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_part));
	HANDLE_ERROR(cudaEventSynchronize(stop_part));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_part, stop_part));

	if (rec_time) t_mult_t += elapsedTime;
#endif

	// update g2, lam2, lam4
	mat_device<Type> Vx(data.p, data.q, data.r, false);
	mat_device<Type> Vy(data.p, data.q, data.r, false);
	mat_device<Type> Vz(data.p, data.q, data.r, false);
	// Vx = Ux - Wx
	vec_add_gpu(data.n, 1, data.Ux.data_dev_ptr(), (Type)(-1.0),
			data.Wx.data_dev_ptr(), Vx.data_dev_ptr(), stream3);
	// Vy = Uy - Wy
	vec_add_gpu(data.n, 1, data.Uy.data_dev_ptr(), (Type)(-1.0),
			data.Wy.data_dev_ptr(), Vy.data_dev_ptr(), stream3);
	// Vz = Uz - Wz
	vec_add_gpu(data.n, 1, data.Uz.data_dev_ptr(), (Type)(-1.0),
	            data.Wz.data_dev_ptr(), Vz.data_dev_ptr(), stream3);

	cudaEvent_t waithere;
	HANDLE_ERROR(cudaEventCreate(&waithere));
	HANDLE_ERROR(cudaEventRecord(waithere, stream1));
	HANDLE_ERROR(cudaStreamWaitEvent(stream2, waithere, 0));
	HANDLE_ERROR(cudaEventDestroy(waithere));

	lam3_5_kernel_wrapper(blocks_lam3_5, threads,
	                      threads * 2 * sizeof(Type), stream2,
	                      data.m, data.Au.data_dev_ptr(),
	                      b.data_dev_ptr(),
	                      data.delta.data_dev_ptr(),
	                      data.gpu_buff3.data_dev_ptr(),
	                      data.gpu_buff4.data_dev_ptr());

	lam2_4_kernel_wrapper(blocks_lam2_4, threads,
	                      threads * 2 * sizeof(Type), stream3,
	                      data.n, Vx.data_dev_ptr(),
	                      Vy.data_dev_ptr(), Vz.data_dev_ptr(),
	                      data.sigmaX.data_dev_ptr(),
	                      data.sigmaZ.data_dev_ptr(),
	                      data.sigmaY.data_dev_ptr(),
	                      data.gpu_buff1.data_dev_ptr(),
	                      data.gpu_buff2.data_dev_ptr());

	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff3.data(),
	                             data.gpu_buff3.data_dev_ptr(),
	                             blocks_lam3_5 * sizeof(Type),
	                             cudaMemcpyDeviceToHost, stream2));
	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff4.data(),
	                             data.gpu_buff4.data_dev_ptr(),
	                             blocks_lam3_5 * sizeof(Type),
	                             cudaMemcpyDeviceToHost, stream2));

	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff1.data(),
	                             data.gpu_buff1.data_dev_ptr(),
	                             blocks_lam2_4 * sizeof(Type),
	                             cudaMemcpyDeviceToHost, stream3));
	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff2.data(),
	                             data.gpu_buff2.data_dev_ptr(),
	                             blocks_lam2_4 * sizeof(Type),
	                             cudaMemcpyDeviceToHost, stream3));

	Dt_gpu(data.g2, Vx, Vy, Vz, stream2);

	HANDLE_ERROR(cudaStreamSynchronize(stream2));

	Type sum3 = 0, sum4 = 0;

	for (int i = 0; i < blocks_lam3_5; i++) {
		sum3 += data.host_buff3[i];
		sum4 += data.host_buff4[i];
	}

	data.lam3 = sum3;
	data.lam5 = sum4;

	HANDLE_ERROR(cudaStreamSynchronize(stream3));

	Type sum1 = 0, sum2 = 0;

	for (int i = 0; i < blocks_lam2_4; i++) {
		sum1 += data.host_buff1[i];
		sum2 += data.host_buff2[i];
	}

	data.lam2 = sum1;
	data.lam4 = sum2;

	HANDLE_ERROR(cudaStreamSynchronize(stream1));

	// g = g - Atb
	HANDLE_ERROR(cublas_axpy(cb_handle, data.n, &val_alpha,
	                         data.Atb.data_dev_ptr(), 1,
	                         data.g.data_dev_ptr(), 1));

	data.f =
	        (data.lam1 + data.beta / 2 * data.lam2 + data.mu / 2 *
	         data.lam3) - data.lam4 - data.lam5;
}

//-----------------------------------------------------------------------------
// get_tau_gpu
//-----------------------------------------------------------------------------
template<class matrix_type, class Type>
Type get_tau_gpu(tval3_data_gpu<Type> &data, const matrix_type &A,
                 bool fst_iter) {
	Type tau;

	// calculate tau
	if (fst_iter) {
		mat_device<Type> dx(data.p, data.q, data.r, false);
		mat_device<Type> dy(data.p, data.q, data.r, false);
		mat_device<Type> dz(data.p, data.q, data.r, false);
		D_gpu(dx, dy, dz, data.d);

		Type result1, result2, result3;
		HANDLE_ERROR(cublas_dot(cb_handle, data.n, dx.data_dev_ptr(), 1,
		                        dx.data_dev_ptr(), 1, &result1));
		HANDLE_ERROR(cublas_dot(cb_handle, data.n, dy.data_dev_ptr(), 1,
		                        dy.data_dev_ptr(), 1, &result2));
		HANDLE_ERROR(cublas_dot(cb_handle, data.n, dz.data_dev_ptr(), 1,
		                        dz.data_dev_ptr(), 1, &result3));
		Type dDd = result1 + result2 + result3;

		Type dd;
		HANDLE_ERROR(cublas_dot(cb_handle, data.n,
		                        data.d.data_dev_ptr(), 1,
		                        data.d.data_dev_ptr(), 1, &dd));

		mat_device<Type> Ad(data.m, false);

		// Ad = A * d
#ifdef PROFILING
		HANDLE_ERROR(cudaEventRecord(start_part));
#endif
		HANDLE_ERROR(mat_vec_mul(CUBLAS_OP_N, A, data.d.data_dev_ptr(),
		                         Ad.data_dev_ptr()));
#ifdef PROFILING
		HANDLE_ERROR(cudaEventRecord(stop_part));
		float elapsedTime;
		HANDLE_ERROR(cudaEventSynchronize(stop_part));
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_part,
		                                  stop_part));

		if (rec_time) t_mult_gettau += elapsedTime;
#endif

		Type Add;
		HANDLE_ERROR(cublas_dot(cb_handle, data.m, Ad.data_dev_ptr(), 1,
		                        Ad.data_dev_ptr(), 1, &Add));

		tau = abs(dd / (dDd + data.muDbeta * Add));
	} else {

		// set block- and grid dimensions for get_tau_kernel
		int threads = 128;
		int blocks = (majorRevision >= 2) ? 64 : 90;
		blocks = min(blocks, (data.n + threads - 1) / threads);

		get_tau_kernel_wrapper(blocks, threads, 2 * threads *
		                       sizeof(Type), 0,
		                       data.n,
		                       data.g.data_dev_ptr(),
		                       data.gp.data_dev_ptr(),
		                       data.g2.data_dev_ptr(),
		                       data.g2p.data_dev_ptr(),
		                       data.uup.data_dev_ptr(),
		                       data.muDbeta,
		                       data.dg.data_dev_ptr(),
		                       data.dg2.data_dev_ptr(),
		                       data.gpu_buff1.data_dev_ptr(),
		                       data.gpu_buff2.data_dev_ptr());

		HANDLE_ERROR(cudaMemcpy(data.host_buff1.data(),
		                        data.gpu_buff1.data_dev_ptr(), blocks *
		                        sizeof(Type),
		                        cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(data.host_buff2.data(),
		                        data.gpu_buff2.data_dev_ptr(), blocks *
		                        sizeof(Type),
		                        cudaMemcpyDeviceToHost));

		Type ss = 0, sy = 0;

		for (int i = 0; i < blocks; i++) {
			ss += data.host_buff1[i];
			sy += data.host_buff2[i];
		}

		tau = abs(ss / sy);
	}
	return tau;
}

//-----------------------------------------------------------------------------
// descend_gpu
//-----------------------------------------------------------------------------
template<class matrix_type, class Type>
void descend_gpu(mat_device<Type> &U, tval3_data_gpu<Type> &data,
                 const matrix_type &A, const mat_device<Type> &b, Type tau,
                 bool nonneg, bool force_mul = true ) {

	// set block- and grid- dimensions for nonneg-kernel
	int threads, blocks;

	if (majorRevision >= 2) {
		threads = 256;
		blocks =
		        max((int)round((double)data.n / 256 / 5 / 16) * 16, 16);
	} else {
		threads = 128;
		blocks = 240;
	}
	blocks = min(blocks, (data.n + threads - 1) / threads);

	Type val_alpha = -1 * tau;
	// U = U - tau*d
	HANDLE_ERROR(cublas_axpy(cb_handle, data.n, &val_alpha,
				data.d.data_dev_ptr(), 1, U.data_dev_ptr(), 1));

	if (nonneg)
		nonneg_kernel_wrapper(blocks, threads, 0, 0,
		                      data.n, U.data_dev_ptr());

	D_gpu(data.Ux, data.Uy, data.Uz, U);

	get_g_gpu(data, A, U, b, force_mul);
}

//-----------------------------------------------------------------------------
// update_g_gpu
//-----------------------------------------------------------------------------
template<class matrix_type, class Type>
void update_g_gpu(tval3_data_gpu<Type> &data, const Type alpha,
                  const matrix_type &A, mat_device<Type> &U,
                  const mat_device<Type> &b) {

	// set block- and grid dimensions for lam2_4_kernel and lam3_5_kernel
	int threads = 128;
	int blocks = (majorRevision >= 2) ? 96 : 90;
	int blocks_lam2_4 = min(blocks, (data.n + threads - 1) / threads);
	int blocks_lam3_5 = min(blocks, (data.m + threads - 1) / threads);

	vec_add_gpu(data.n, 1,
	            data.gp.data_dev_ptr(), alpha,
	            data.dg.data_dev_ptr(), data.g.data_dev_ptr(), stream1);
	vec_add_gpu(data.n, 1,
	            data.g2p.data_dev_ptr(), alpha,
	            data.dg2.data_dev_ptr(), data.g2.data_dev_ptr(), stream2);
	vec_add_gpu(data.n, 1,
	            data.Up.data_dev_ptr(), alpha,
	            data.dU.data_dev_ptr(), U.data_dev_ptr(), stream3);
	vec_add_gpu(data.m, 1,
	            data.Aup.data_dev_ptr(), alpha,
	            data.dAu.data_dev_ptr(), data.Au.data_dev_ptr(), stream1);
	vec_add_gpu(data.n, 1,
	            data.Uxp.data_dev_ptr(), alpha,
	            data.dUx.data_dev_ptr(), data.Ux.data_dev_ptr(), stream2);
	vec_add_gpu(data.n, 1,
	            data.Uyp.data_dev_ptr(), alpha,
	            data.dUy.data_dev_ptr(), data.Uy.data_dev_ptr(), stream3);
	vec_add_gpu(data.n, 1,
	            data.Uzp.data_dev_ptr(), alpha,
	            data.dUz.data_dev_ptr(), data.Uz.data_dev_ptr(), stream3);

	// update lam2, lam4
	mat_device<Type> Vx(data.p, data.q, data.r, false);
	mat_device<Type> Vy(data.p, data.q, data.r, false);
	mat_device<Type> Vz(data.p, data.q, data.r, false);
	// Vx = Ux - Wx
	vec_add_gpu(data.n, 1, data.Ux.data_dev_ptr(), (Type)(-1.0),
	            data.Wx.data_dev_ptr(), Vx.data_dev_ptr(), stream2);
	// Vy = Uy - Wy
	vec_add_gpu(data.n, 1, data.Uy.data_dev_ptr(), (Type)(-1.0),
	            data.Wy.data_dev_ptr(), Vy.data_dev_ptr(), stream3);
	// Vz = Uz - Wz
	vec_add_gpu(data.n, 1, data.Uz.data_dev_ptr(), (Type)(-1.0),
	            data.Wz.data_dev_ptr(), Vz.data_dev_ptr(), stream3);

	HANDLE_ERROR(cudaStreamSynchronize(stream1));

	lam3_5_kernel_wrapper(blocks_lam3_5, threads,
	                      threads * 2 * sizeof(Type), stream2,
	                      data.m, data.Au.data_dev_ptr(),
	                      b.data_dev_ptr(),
	                      data.delta.data_dev_ptr(),
	                      data.gpu_buff3.data_dev_ptr(),
	                      data.gpu_buff4.data_dev_ptr());

	lam2_4_kernel_wrapper(blocks_lam2_4, threads,
	                      threads * 2 * sizeof(Type), stream3,
	                      data.n, Vx.data_dev_ptr(),
	                      Vy.data_dev_ptr(), Vz.data_dev_ptr(),
	                      data.sigmaX.data_dev_ptr(),
	                      data.sigmaY.data_dev_ptr(),
	                      data.sigmaZ.data_dev_ptr(),
	                      data.gpu_buff1.data_dev_ptr(),
	                      data.gpu_buff2.data_dev_ptr());

	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff3.data(),
	                             data.gpu_buff3.data_dev_ptr(),
	                             blocks_lam3_5 * sizeof(Type),
	                             cudaMemcpyDeviceToHost, stream2));
	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff4.data(),
	                             data.gpu_buff4.data_dev_ptr(),
	                             blocks_lam3_5 * sizeof(Type),
	                             cudaMemcpyDeviceToHost, stream2));

	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff1.data(),
	                             data.gpu_buff1.data_dev_ptr(),
	                             blocks_lam2_4 * sizeof(Type),
	                             cudaMemcpyDeviceToHost, stream3));
	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff2.data(),
	                             data.gpu_buff2.data_dev_ptr(),
	                             blocks_lam2_4 * sizeof(Type),
	                             cudaMemcpyDeviceToHost, stream3));

	HANDLE_ERROR(cudaStreamSynchronize(stream2));

	Type sum3 = 0, sum4 = 0;

	for (int i = 0; i < blocks_lam3_5; i++) {
		sum3 += data.host_buff3[i];
		sum4 += data.host_buff4[i];
	}

	data.lam3 = sum3;
	data.lam5 = sum4;

	HANDLE_ERROR(cudaStreamSynchronize(stream3));

	Type sum1 = 0, sum2 = 0;

	for (int i = 0; i < blocks_lam2_4; i++) {
		sum1 += data.host_buff1[i];
		sum2 += data.host_buff2[i];
	}

	data.lam2 = sum1;
	data.lam4 = sum2;


	data.f = data.lam1 + data.beta / 2 * data.lam2 + data.mu / 2 *
	         data.lam3 - data.lam4 - data.lam5;
}

//-----------------------------------------------------------------------------
// min_u_gpu
//-----------------------------------------------------------------------------
template<class matrix_type, class Type>
void min_u_gpu(tval3_data_gpu<Type> &data, mat_device<Type> &U, Type &gam,
               const matrix_type &A, const mat_device<Type> &b,
               const tval3_options<Type> &opts, Type C, bool fst_iter) {
	Type tau, alpha, c_armij;
	tau = get_tau_gpu(data, A, fst_iter);

	// keep previous values
	data.Up = U; data.gp = data.g; data.g2p = data.g2;
	data.Aup = data.Au; data.Uxp = data.Ux; data.Uyp = data.Uy;
	data.Uzp = data.Uz;

	// one step steepest descend
	descend_gpu(U, data, A, b, tau, opts.nonneg, false);

	// NMLA
	alpha = 1;
	// dU = U - Up
	vec_add_gpu(data.n, 1, U.data_dev_ptr(), (Type)(-1.0),
			data.Up.data_dev_ptr(), data.dU.data_dev_ptr());
	// c_armij = d'*d
	HANDLE_ERROR(cublas_dot(cb_handle, data.n, data.d.data_dev_ptr(), 1,
				data.d.data_dev_ptr(), 1, &c_armij));
	c_armij *= tau * opts.c * data.beta;

	if (abs(data.f) > abs(C - alpha * c_armij)) {  // Armijo condition

		// dg=g-gp
		vec_add_gpu(data.n, 1, data.g.data_dev_ptr(), (Type)(-1.0),
				data.gp.data_dev_ptr(), data.dg.data_dev_ptr());

		// dg2=g2-g2p
		vec_add_gpu(data.n, 1, data.g2.data_dev_ptr(), (Type)(-1.0),
				data.g2p.data_dev_ptr(), data.dg2.data_dev_ptr());

		// dAu=Au-Aup
		vec_add_gpu(data.m, 1, data.Au.data_dev_ptr(), (Type)(-1.0),
				data.Aup.data_dev_ptr(), data.dAu.data_dev_ptr());

		// dUx=Ux-Uxp
		vec_add_gpu(data.n, 1, data.Ux.data_dev_ptr(), (Type)(-1.0),
				data.Uxp.data_dev_ptr(), data.dUx.data_dev_ptr());

		// dUy = Uy-Uyp
		vec_add_gpu(data.n, 1, data.Uy.data_dev_ptr(), (Type)(-1.0),
				data.Uyp.data_dev_ptr(), data.dUy.data_dev_ptr());

		// dUz = Uz-Uzp
		vec_add_gpu(data.n, 1, data.Uz.data_dev_ptr(), (Type)(-1.0),
				data.Uzp.data_dev_ptr(), data.dUz.data_dev_ptr());

		int cnt = 0;
		while (abs(data.f) > abs(C - alpha * c_armij)) { // Armijo condition
			if (cnt == 5) { // "backtracking" not successful
				gam *= opts.rate_gam;
				tau = get_tau_gpu(data, A, true);
				U = data.Up;
				descend_gpu(U, data, A, b, tau, opts.nonneg,
				            true);
				break;
			}
			alpha *= opts.gamma;

			update_g_gpu(data, alpha, A, U, b);

			cnt++;
		}
	}
}

//-----------------------------------------------------------------------------
// get_gradient_gpu
//-----------------------------------------------------------------------------
template<class Type>
void get_gradient_gpu(tval3_data_gpu<Type> &data,
                      const mat_device<Type> &DtsAtd) {

	// set grid and block dimensions for vec_add_kernel
	int threads, blocks;

	if (majorRevision >= 2) {
		threads = 256;
		blocks =
		        max((int)round((double)data.n / 256 / 2 / 16) * 16, 16);
	} else {
		threads = 128;
		blocks = 240;
	}
	blocks = min(blocks, (data.n + threads - 1) / threads);

	// d = g2 + muDbeta*g + DtsAtd   (DtsAtd has opposite sign, compared to 
	// the MATLAB version)
	vec_add_kernel_wrapper(blocks, threads, 0, 0, data.n,
	                       data.d.data_dev_ptr(), data.muDbeta,
	                       data.g.data_dev_ptr(),
	                       data.g2.data_dev_ptr(), DtsAtd.data_dev_ptr());
}

//-----------------------------------------------------------------------------
// shrinkage_gpu
//-----------------------------------------------------------------------------
template<class Type>
void shrinkage_gpu(tval3_data_gpu<Type> &data) {
	int threads, blocks;

	if (majorRevision >= 2) {
		threads = 128;
		blocks = 64;
	} else {
		threads = 128;
		blocks = 90;
	}
	blocks = min(blocks, (data.n + threads - 1) / threads);

	shrinkage_kernel_wrapper(blocks, threads, threads * sizeof(Type), 0,
	                         data.n,
	                         data.Ux.data_dev_ptr(),
	                         data.Uy.data_dev_ptr(),
	                         data.Uz.data_dev_ptr(),
	                         data.sigmaX.data_dev_ptr(),
	                         data.sigmaY.data_dev_ptr(),
	                         data.sigmaZ.data_dev_ptr(), data.beta,
	                         data.Wx.data_dev_ptr(),
	                         data.Wy.data_dev_ptr(),
	                         data.Wz.data_dev_ptr(),
	                         data.gpu_buff1.data_dev_ptr());

	HANDLE_ERROR(cudaMemcpy(data.host_buff1.data(),
	                        data.gpu_buff1.data_dev_ptr(), blocks *
	                        sizeof(Type),
	                        cudaMemcpyDeviceToHost));

	Type sum = 0;
	for (int i = 0; i < blocks; i++)
		sum += data.host_buff1[i];

	data.lam1 = sum;
}

//-----------------------------------------------------------------------------
// update_W_gpu
//-----------------------------------------------------------------------------
template<class Type>
void update_W_gpu(tval3_data_gpu<Type> &data) {

	// set block- and grid dimensions for lam2_4_kernel
	int threads = 128;
	int blocks = (majorRevision >= 2) ? 96 : 90;
	blocks = min(blocks, (data.n + threads - 1) / threads);

	data.f -= (data.lam1 + data.beta / 2 * data.lam2 - data.lam4);

	shrinkage_gpu(data);

	// update g2, lam2, lam4
	mat_device<Type> Vx(data.p, data.q, data.r, false);
	mat_device<Type> Vy(data.p, data.q, data.r, false);
	mat_device<Type> Vz(data.p, data.q, data.r, false);
	
	// Vx = Ux - Wx
	vec_add_gpu(data.n, 1, data.Ux.data_dev_ptr(), (Type)(-1.0),
			data.Wx.data_dev_ptr(), Vx.data_dev_ptr(), stream1);
	// Vy = Uy - Wy
	vec_add_gpu(data.n, 1, data.Uy.data_dev_ptr(), (Type)(-1.0),
			data.Wy.data_dev_ptr(), Vy.data_dev_ptr(), stream2);
	// Vz = Uz - Wz
	vec_add_gpu(data.n, 1, data.Uz.data_dev_ptr(), (Type)(-1.0),
			data.Wz.data_dev_ptr(), Vz.data_dev_ptr(), stream3);

	HANDLE_ERROR(cudaStreamSynchronize(stream1));
	HANDLE_ERROR(cudaStreamSynchronize(stream2));

	lam2_4_kernel_wrapper(blocks, threads, threads * 2 * sizeof(Type),
	                      stream1,
	                      data.n, Vx.data_dev_ptr(),
	                      Vy.data_dev_ptr(), Vz.data_dev_ptr(),
	                      data.sigmaX.data_dev_ptr(),
	                      data.sigmaY.data_dev_ptr(),
	                      data.sigmaZ.data_dev_ptr(),
	                      data.gpu_buff1.data_dev_ptr(),
	                      data.gpu_buff2.data_dev_ptr());

	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff1.data(),
	                             data.gpu_buff1.data_dev_ptr(), blocks *
	                             sizeof(Type),
	                             cudaMemcpyDeviceToHost, stream1));
	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff2.data(),
	                             data.gpu_buff2.data_dev_ptr(), blocks *
	                             sizeof(Type),
	                             cudaMemcpyDeviceToHost, stream1));

	Dt_gpu(data.g2, Vx, Vy, Vz, stream2);

	Type sum1 = 0, sum2 = 0;

	HANDLE_ERROR(cudaStreamSynchronize(stream1));

	for (int i = 0; i < blocks; i++) {
		sum1 += data.host_buff1[i];
		sum2 += data.host_buff2[i];
	}

	data.lam2 = sum1;
	data.lam4 = sum2;

	data.f += (data.lam1 + data.beta / 2 * data.lam2 - data.lam4);
}

//-----------------------------------------------------------------------------
// update_mlp_gpu
//-----------------------------------------------------------------------------
template<class Type>
void update_mlp_gpu(tval3_data_gpu<Type> &data, const mat_device<Type> &b) {
	data.f += (data.lam4 + data.lam5);

	mat_device<Type> Vx(data.p, data.q, data.r, false);
	mat_device<Type> Vy(data.p, data.q, data.r, false);
	mat_device<Type> Vz(data.p, data.q, data.r, false);
	// Vx = Ux - Wx
	vec_add_gpu(data.n, 1, data.Ux.data_dev_ptr(), (Type)(-1.0),
			data.Wx.data_dev_ptr(), Vx.data_dev_ptr());
	// Vy = Uy - Wy
	vec_add_gpu(data.n, 1, data.Uy.data_dev_ptr(), (Type)(-1.0),
			data.Wy.data_dev_ptr(), Vy.data_dev_ptr());
	// Vz = Uz - Wz
	vec_add_gpu(data.n, 1, data.Uz.data_dev_ptr(), (Type)(-1.0),
			data.Wz.data_dev_ptr(), Vz.data_dev_ptr());

	Type val_alpha = -1 * data.beta;
	// sigmaX -= beta*Vx
	HANDLE_ERROR(cublas_axpy(cb_handle, data.n, &val_alpha,
				Vx.data_dev_ptr(), 1, data.sigmaX.data_dev_ptr(), 1));
	// sigmaY -= beta*Vy
	HANDLE_ERROR(cublas_axpy(cb_handle, data.n, &val_alpha,
				Vy.data_dev_ptr(), 1, data.sigmaY.data_dev_ptr(), 1));
	// sigmaZ -= beta*Vz
	HANDLE_ERROR(cublas_axpy(cb_handle, data.n, &val_alpha,
				Vz.data_dev_ptr(), 1, data.sigmaZ.data_dev_ptr(), 1));

	Type result1, result2;
	HANDLE_ERROR(cublas_dot(cb_handle, data.n, data.sigmaX.data_dev_ptr(),
				1, Vx.data_dev_ptr(), 1, &data.lam4));
	HANDLE_ERROR(cublas_dot(cb_handle, data.n, data.sigmaY.data_dev_ptr(),
				1, Vy.data_dev_ptr(), 1, &result1));
	HANDLE_ERROR(cublas_dot(cb_handle, data.n, data.sigmaZ.data_dev_ptr(),
				1, Vz.data_dev_ptr(), 1, &result2));
	data.lam4 += result1 + result2;

	mat_device<Type> Aub(data.m, false);
	// Aub = Au - b
	vec_add_gpu(data.m, 1, data.Au.data_dev_ptr(), (Type)(-1.0),
			b.data_dev_ptr(), Aub.data_dev_ptr());
	val_alpha = -1 * data.mu;
	// delta -= mu*Aub
	HANDLE_ERROR(cublas_axpy(cb_handle, data.m, &val_alpha,
				Aub.data_dev_ptr(), 1, data.delta.data_dev_ptr(), 1));
	HANDLE_ERROR(cublas_dot(cb_handle, data.m, data.delta.data_dev_ptr(), 1,
	                        Aub.data_dev_ptr(), 1, &data.lam5));

	data.f -= (data.lam4 + data.lam5);
}

//-----------------------------------------------------------------------------
// tval3_gpu_3d: main function
//-----------------------------------------------------------------------------
template<class matrix_type, class Type>
const tval3_info<Type> tval3_gpu_3d(mat_host<Type> &U, const matrix_type &A,
		const mat_host<Type> &b, const tval3_options<Type> &opts,
		const mat_host<Type> &Ut, bool pagelocked) {

	float elapsedTime;
	cudaEvent_t start_tval3, stop_tval3;
	HANDLE_ERROR(cudaEventCreate(&start_tval3));
	HANDLE_ERROR(cudaEventCreate(&stop_tval3));
#ifdef PROFILING
	cudaEvent_t start_loop, stop_loop;
	HANDLE_ERROR(cudaEventCreate(&start_loop));
	HANDLE_ERROR(cudaEventCreate(&stop_loop));
	HANDLE_ERROR(cudaEventCreate(&start_part));
	HANDLE_ERROR(cudaEventCreate(&stop_part));
	t_mult_t = 0; t_mult_nt = 0; t_mult_gettau = 0; t_dt = 0; t_d = 0;
#endif

	printf("---------------------------------- processing ----------------------------------\n");

	cudaEventRecord (start_tval3);

	if (majorRevision >= 2) {
		HANDLE_ERROR(cudaStreamCreate(&stream1));
		HANDLE_ERROR(cudaStreamCreate(&stream2));
		HANDLE_ERROR(cudaStreamCreate(&stream3));
	} else {
		stream1 = 0;
		stream2 = 0;
		stream3 = 0;
	}

	tval3_data_gpu<Type> data(U, b, opts, pagelocked);

	mat_device<Type> b_gpu = b;

	Type scl = scaleb_gpu(b, b_gpu);

	mat_device<Type> DtsAtd(data.n);
	// Atb = A'*b
	HANDLE_ERROR(mat_vec_mul(CUBLAS_OP_T, A, b_gpu.data_dev_ptr(),
	                         data.Atb.data_dev_ptr()));

	mat_device<Type> U_gpu(data.p, data.q, data.r, false, false);
	U_gpu = data.Atb; // initial guess: U=A'*b

	Type muf = opts.mu;
	Type betaf = opts.beta;
	Type beta0 = 0;
	Type gam = opts.gam;

	if (data.mu > muf) data.mu = muf;

	if (data.beta > betaf) data.beta = betaf;
	data.muDbeta = data.mu / data.beta;

	mat_device<Type> rcdU = U_gpu, UrcdU(data.n, false);

	D_gpu(data.Ux, data.Uy, data.Uz, U_gpu);
	shrinkage_gpu(data);
	get_g_gpu(data, A, U_gpu, b_gpu, true);
	get_gradient_gpu(data, DtsAtd);

	Type Q = 1, Qp;
	Type C = data.f;
	int count_outer = 0, count_total = 0;
	bool fst_iter;
	Type RelChg, RelChgOut = 0, nrmup = 0;

#ifdef PROFILING
	rec_time = true;
	cudaEventRecord (start_loop);
#endif

	while ((count_outer < opts.maxcnt) && (count_total < opts.maxit)) {

		cout << "count_outer = " << count_outer << endl;

		fst_iter = true;
		while (count_total < opts.maxit) {
			// u-subproblem
			min_u_gpu(data, U_gpu, gam, A, b_gpu, opts, C,
			          fst_iter);

			// shrinkage like step
			update_W_gpu(data);

			// update reference values
			Qp = Q, Q = gam * Qp + 1; C =
			        (gam * Qp * C + data.f) / Q;
			vec_add_gpu(data.n, 1,
			            U_gpu.data_dev_ptr(), (Type)(-1.0),
			            data.Up.data_dev_ptr(),
			            data.uup.data_dev_ptr());

			// compute gradient
			get_gradient_gpu(data, DtsAtd);

			count_total++;

			// calculate relative change
			HANDLE_ERROR(cublas_nrm2(cb_handle, data.n,
			                         data.Up.data_dev_ptr(), 1,
			                         &nrmup));


			HANDLE_ERROR(cublas_nrm2(cb_handle, data.n,
			                         data.uup.data_dev_ptr(), 1,
			                         &RelChg));
			RelChg /= nrmup;

			if ((RelChg < opts.tol_inn) && (!fst_iter))
				break;

			fst_iter = false;

		}

		count_outer++;

		// calculate relative change
		// UrcdU = U - rcdU
		vec_add_gpu(data.n, 1, U_gpu.data_dev_ptr(), (Type)(-1.0),
				rcdU.data_dev_ptr(), UrcdU.data_dev_ptr());

		HANDLE_ERROR(cublas_nrm2(cb_handle, data.n, UrcdU.data_dev_ptr(), 1,
					&RelChgOut));
		RelChgOut /= nrmup;

		rcdU = U_gpu;

		if (RelChgOut < opts.tol)
			break;

		// update multipliers
		update_mlp_gpu(data, b_gpu);

		// update penalty parameters for continuation scheme
		beta0 = data.beta;
		data.beta *= opts.rate_cnt;
		data.mu *= opts.rate_cnt;

		if (data.beta > betaf) data.beta = betaf;

		if (data.mu > muf) data.mu = muf;
		data.muDbeta = data.mu / data.beta;

		// update f
		data.f = data.lam1 + data.beta / 2 * data.lam2 + data.mu / 2 *
		         data.lam3 - data.lam4 - data.lam5;

		// DtsAtd = beta0/beta*d
		DtsAtd = data.d;
		Type val_alpha = beta0 / data.beta;
		HANDLE_ERROR(cublas_scal(cb_handle, data.n, &val_alpha,
		                         DtsAtd.data_dev_ptr(), 1));

		// compute gradient
		get_gradient_gpu(data, DtsAtd);

		// reset reference values
		gam = opts.gam; Q = 1; C = data.f;

	}

#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_loop));
	rec_time = false;
	HANDLE_ERROR(cudaEventSynchronize(stop_loop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_loop, stop_loop));

	/*
	   ofstream file;
	   file.open("profile_gpu.txt");
	   file << "loop [us]:\t" << elapsedTime << "\n";
	   file << "mult (not trans.) [us]:\t" << t_mult_nt << "\n";
	   file << "mult (trans.) [us]:\t" << t_mult_t << "\n";
	   file << "mult (in get_tau()) [us]:\t" << t_mult_gettau << "\n";
	   file << "d [us]:\t" << t_d << "\n";
	   file << "dt [us]:\t" << t_dt << "\n";
	   file.close();*/
#endif

	// copy result to host
	mat_gpu_to_host(U, U_gpu);

	// scale U.
	for (int i = 0; i < data.n; i++) {
		U[i] = U[i] / scl;
	}

	// copy U to device
	U_gpu = U;

	HANDLE_ERROR(cudaEventRecord (stop_tval3));
	HANDLE_ERROR(cudaEventSynchronize(stop_tval3));

#ifdef PROFILING

	printf("---------------------------------- profiling ----------------------------------\n");

	for (int i = 0; i < 100; i++) {

		if (profile_info[i].valid == true)
			printf("%s:\t %.2f ms (runs: %i, total: %.1f ms)\n",
					profile_info[i].name, profile_info[i].time / profile_info[i].runs,
					profile_info[i].runs, profile_info[i].time);
	}

#endif

	// generate return value
	tval3_info<Type> info;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_tval3,
	                                  stop_tval3));
	info.secs = elapsedTime / 1000;
	info.outer_iters = count_outer;
	info.total_iters = count_total;
	info.rel_chg = RelChgOut;

	if (Ut.len == data.n) {
		mat_device<Type> Ut_gpu = Ut;
		mat_device<Type> deltaU = Ut_gpu;
		Type val_alpha = -1;
		HANDLE_ERROR(cublas_axpy(cb_handle, data.n, &val_alpha,
		                         U_gpu.data_dev_ptr(), 1,
		                         deltaU.data_dev_ptr(), 1));
		Type nrm_deltaU;
		HANDLE_ERROR(cublas_nrm2(cb_handle, data.n,
		                         deltaU.data_dev_ptr(), 1,
		                         &nrm_deltaU));
		Type nrm_Ut;
		HANDLE_ERROR(cublas_nrm2(cb_handle, data.n,
		                         Ut_gpu.data_dev_ptr(), 1, &nrm_Ut));
		info.rel_error = nrm_deltaU / nrm_Ut;
	} else
		info.rel_error = 0;

	if (majorRevision >= 2) {
		HANDLE_ERROR(cudaStreamDestroy(stream1));
		HANDLE_ERROR(cudaStreamDestroy(stream2));
		HANDLE_ERROR(cudaStreamDestroy(stream3));
	}

	HANDLE_ERROR(cudaEventDestroy(start_tval3));
	HANDLE_ERROR(cudaEventDestroy(stop_tval3));
#ifdef PROFILING
	HANDLE_ERROR(cudaEventDestroy(start_loop));
	HANDLE_ERROR(cudaEventDestroy(stop_loop));
	HANDLE_ERROR(cudaEventDestroy(start_part));
	HANDLE_ERROR(cudaEventDestroy(stop_part));
#endif

	printf(
	        "----------------------------------------------------------------------------------\n");

	return info;
}

//-----------------------------------------------------------------------------
// check_params_1
//-----------------------------------------------------------------------------
template<class Type>
void check_params_1(const mat_host<Type> &U,
                    const mat_host<Type> &Ut) throw(tval3_gpu_exception) {
	// Do checks only if Ut exists  not possible for real data
	if ( &Ut != NULL ) {
		if (U.format != mat_col_major || Ut.format != mat_col_major)
			throw tval3_gpu_exception(
			              "Argument error: Matrices must be in column major format!");

		else if (Ut.len > 0 && Ut.len != U.len)
			throw tval3_gpu_exception(
			              "Argument error: U and Ut must have the same size!");

	}
}

//-----------------------------------------------------------------------------
// check_params_2
//-----------------------------------------------------------------------------
template<class mat_Type, class Type>
void check_params_2(const mat_host<Type> &U, const mat_Type &A,
                    const mat_host<Type> &b) throw(tval3_gpu_exception) {
	if (U.len != A.dim_x)
		throw tval3_gpu_exception(
		              "Argument error: the length of U must be equal to A.dim_x!");

	else if (b.len != A.dim_y)
		throw tval3_gpu_exception(
		              "Argument error: b.len must be equal to A.dim_y!");

}

//-----------------------------------------------------------------------------
// getMajorRevision
//-----------------------------------------------------------------------------
int getMajorRevision(int device) {
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
	return prop.major;
}

//-----------------------------------------------------------------------------
// tval3_gpu_3d overloaded for dense matrixes (mat_host)
//-----------------------------------------------------------------------------
template<class Type>
const tval3_info<Type> tval3_gpu_3d(mat_host<Type> &U, const mat_host<Type> &A,
		const mat_host<Type> &b, const tval3_options<Type> &opts,
		const mat_host<Type> &Ut, bool pagelocked, int mainGPU, int numGPUs, 
		int *gpuArray ) {

	int cudaDevice;
	tval3_info<Type> info;

	try {
		if (A.format != mat_col_major)
			throw tval3_gpu_exception(
			              "Argument error: A must be in column major format!");

		check_params_1(U, Ut);
		check_params_2(U, A, b);

		HANDLE_ERROR(cudaGetDevice(&cudaDevice));
		majorRevision = getMajorRevision(cudaDevice);

		HANDLE_ERROR(cublasCreate(&cb_handle));
#ifdef PROFILING
		HANDLE_ERROR(cudaEventCreate(&start));
		HANDLE_ERROR(cudaEventCreate(&stop));
		HANDLE_ERROR(cudaEventCreate(&start_cu));
		HANDLE_ERROR(cudaEventCreate(&stop_cu));
#endif

		if ((mainGPU != -1) | (numGPUs != 0) | (gpuArray != NULL)) {
			printf(
			        "INFO: Multi-GPU settings do not effect this version!");
		}

		dense_mm<Type> mA(A, cb_handle);

		info = tval3_gpu_3d(U, mA, b, opts, Ut, pagelocked);

		HANDLE_ERROR(cublasDestroy(cb_handle));

#ifdef PROFILING
		HANDLE_ERROR(cudaEventDestroy(start));
		HANDLE_ERROR(cudaEventDestroy(stop));
		HANDLE_ERROR(cudaEventDestroy(start_cu));
		HANDLE_ERROR(cudaEventDestroy(stop_cu));
#endif
	} catch (tval3_gpu_exception) {
		throw;
	} catch (const std::exception &ex) {
		std::string msg = "Internal error: ";
		msg += ex.what();
		throw tval3_gpu_exception(msg);
	}

	return info;
}

//-----------------------------------------------------------------------------
// tval3_gpu_3d overloaded for sparse matrixes (sparse_mat_host)
//-----------------------------------------------------------------------------
template<class Type>
const tval3_info<Type> tval3_gpu_3d(mat_host<Type> &U,
		const sparse_mat_host<Type> &A, const mat_host<Type> &b,
		const tval3_options<Type> &opts, const mat_host<Type> &Ut, bool pagelocked,
		int mainGPU, int numGPUs, int *gpuArray ) {

	int cudaDevice;
	tval3_info<Type> info;

	try {
		if (A.format != sparse_mat_both)
			throw tval3_gpu_exception(
					"Argument error: A must be available in CSR and CSC format!");

		check_params_1(U, Ut);
		check_params_2(U, A, b);

		HANDLE_ERROR(cudaGetDevice(&cudaDevice));
		majorRevision = getMajorRevision(cudaDevice);

		HANDLE_ERROR(cublasCreate(&cb_handle));

#ifdef PROFILING
		HANDLE_ERROR(cudaEventCreate(&start));
		HANDLE_ERROR(cudaEventCreate(&stop));
		HANDLE_ERROR(cudaEventCreate(&start_cu));
		HANDLE_ERROR(cudaEventCreate(&stop_cu));
#endif

		double difftime;
		timeval start0, stop0;

		gettimeofday(&start0, NULL);

		if ( mainGPU == -1)
			mainGPU = 0;

		if ((numGPUs == 0) | (gpuArray == NULL)) {
			numGPUs = 1;
			gpuArray = new int[numGPUs];
			gpuArray[0] = mainGPU;
		}

		// multi-threaded initialization process: wasn't faster than other!
		//sparse_mm_multGPU_threaded mA1(A, numGPUs, gpuArray, mainGPU); 
		sparse_mm_multGPU<Type> mA1(A, numGPUs, gpuArray, mainGPU);
		info = tval3_gpu_3d(U, mA1, b, opts, Ut, pagelocked);

		gettimeofday(&stop0, NULL);
		difftime =
		        (double)((stop0.tv_sec * 1000000 +
		                  stop0.tv_usec) -
		                 (start0.tv_sec * 1000000 + start0.tv_usec));

		info.outer_secs = difftime / (double)1e6;

#ifdef PROFILING
		HANDLE_ERROR(cudaEventDestroy(start));
		HANDLE_ERROR(cudaEventDestroy(stop));
		HANDLE_ERROR(cudaEventDestroy(start_cu));
		HANDLE_ERROR(cudaEventDestroy(stop_cu));
#endif

		HANDLE_ERROR(cublasDestroy(cb_handle));

	} catch (tval3_gpu_exception) {
		throw;
	} catch (const std::exception &ex) {
		std::string msg = "Internal error: ";
		msg += ex.what();
		throw tval3_gpu_exception(msg);
	}

	return info;
}

//-----------------------------------------------------------------------------
// tval3_gpu_3d overloaded for dynamic computation (geometry_host)
//-----------------------------------------------------------------------------
template<class Type>
const tval3_info<Type> tval3_gpu_3d(mat_host<Type> &U, const geometry_host &A,
		const mat_host<Type> &b, const tval3_options<Type> &opts,
		const mat_host<Type> &Ut, bool pagelocked, int mainGPU, int numGPUs, 
		int *gpuArray ) {

	int cudaDevice;
	tval3_info<Type> info;

	try {
		check_params_1(U, Ut);

		HANDLE_ERROR(cudaGetDevice(&cudaDevice));
		majorRevision = getMajorRevision(cudaDevice);

		HANDLE_ERROR(cublasCreate(&cb_handle));

#ifdef PROFILING
		HANDLE_ERROR(cudaEventCreate(&start));
		HANDLE_ERROR(cudaEventCreate(&stop));
		HANDLE_ERROR(cudaEventCreate(&start_cu));
		HANDLE_ERROR(cudaEventCreate(&stop_cu));
#endif

		double difftime;
		timeval start0, stop0;

		gettimeofday(&start0, NULL);

		if ( mainGPU == -1)
			mainGPU = 0;

		if ((numGPUs == 0) | (gpuArray == NULL)) {
			numGPUs = 1;
			gpuArray = new int[numGPUs];
			gpuArray[0] = mainGPU;
		}

		geom_mm_multGPU<Type> mA(A, numGPUs, gpuArray, mainGPU, 16);
		info = tval3_gpu_3d(U, mA, b, opts, Ut, pagelocked);

		gettimeofday(&stop0, NULL);
		difftime =
		        (double)((stop0.tv_sec * 1000000 +
		                  stop0.tv_usec) -
		                 (start0.tv_sec * 1000000 + start0.tv_usec));

		info.outer_secs = difftime / (double)1e6;

#ifdef PROFILING
		HANDLE_ERROR(cudaEventDestroy(start));
		HANDLE_ERROR(cudaEventDestroy(stop));
		HANDLE_ERROR(cudaEventDestroy(start_cu));
		HANDLE_ERROR(cudaEventDestroy(stop_cu));
#endif
		HANDLE_ERROR(cublasDestroy(cb_handle));
	} catch (tval3_gpu_exception) {
		throw;
	} catch (const std::exception &ex) {
		std::string msg = "Internal error: ";
		msg += ex.what();
		throw tval3_gpu_exception(msg);
	}

	return info;
}

//-----------------------------------------------------------------------------
// tval3_gpu_3d overloaded for combined dyn-calcN / sparseT (different 
// signature than calls above)
//-----------------------------------------------------------------------------
template<class Type>
const tval3_info<Type> tval3_gpu_3d(mat_host<Type> &U,
		const geometry_host &A_geo, const sparse_mat_host<Type> &A_mat,
		const mat_host<Type> &b, const tval3_options<Type> &opts,
		const mat_host<Type> &Ut, bool pagelocked, int mainGPU, int numGPUs, 
		int *gpuArray ) {

	int cudaDevice;
	tval3_info<Type> info;

	try {
		check_params_1(U, Ut);

		HANDLE_ERROR(cudaGetDevice(&cudaDevice));
		majorRevision = getMajorRevision(cudaDevice);

		HANDLE_ERROR(cublasCreate(&cb_handle));

#ifdef PROFILING
		HANDLE_ERROR(cudaEventCreate(&start));
		HANDLE_ERROR(cudaEventCreate(&stop));
		HANDLE_ERROR(cudaEventCreate(&start_cu));
		HANDLE_ERROR(cudaEventCreate(&stop_cu));
#endif

		double difftime;
		timeval start0, stop0;

		gettimeofday(&start0, NULL);

		if ( mainGPU == -1)
			exit(-1);

		if ((numGPUs == 0) | (gpuArray == NULL)) {
			numGPUs = 1;
			gpuArray = new int[numGPUs];
			gpuArray[0] = mainGPU;
		}

		sparse_geom_mm<Type> mA(A_geo, A_mat, numGPUs, gpuArray,
		                        mainGPU);
		info = tval3_gpu_3d(U, mA, b, opts, Ut, pagelocked);

		gettimeofday(&stop0, NULL);
		difftime = (double)((stop0.tv_sec * 1000000 + stop0.tv_usec) -
		                 (start0.tv_sec * 1000000 + start0.tv_usec));

		info.outer_secs = difftime / (double)1e6;

#ifdef PROFILING
		HANDLE_ERROR(cudaEventDestroy(start));
		HANDLE_ERROR(cudaEventDestroy(stop));
		HANDLE_ERROR(cudaEventDestroy(start_cu));
		HANDLE_ERROR(cudaEventDestroy(stop_cu));
#endif
		HANDLE_ERROR(cublasDestroy(cb_handle));
	} catch (tval3_gpu_exception) {
		throw;
	} catch (const std::exception &ex) {
		std::string msg = "Internal error: ";
		msg += ex.what();
		throw tval3_gpu_exception(msg);
	}

	return info;
}

//-----------------------------------------------------------------------------
// Type resolved function calls (callable from outside)
//-----------------------------------------------------------------------------
const tval3_info<float> tval3_gpu_3d(mat_host<float> &U,
		const sparse_mat_host<float> &A, const mat_host<float> &b,
		const tval3_options<float> &opts, const mat_host<float> &Ut, 
		bool pagelocked, int mainGPU, int numGPUs, int *gpuArray ) {

	return tval3_gpu_3d<float>(U, A, b, opts, Ut, pagelocked, mainGPU,
	                           numGPUs, gpuArray );
}

const tval3_info<float> tval3_gpu_3d(mat_host<float> &U,
		const mat_host<float> &A, const mat_host<float> &b,
		const tval3_options<float> &opts, const mat_host<float> &Ut, 
		bool pagelocked, int mainGPU, int numGPUs, int *gpuArray ) {

	return tval3_gpu_3d<float>(U, A, b, opts, Ut, pagelocked, mainGPU,
	                           numGPUs, gpuArray );
}

const tval3_info<float> tval3_gpu_3d(mat_host<float> &U, const geometry_host &A,
		const mat_host<float> &b, const tval3_options<float> &opts,
		const mat_host<float> &Ut, bool pagelocked, int mainGPU, int numGPUs, 
		int *gpuArray ) {

	return tval3_gpu_3d<float>(U, A, b, opts, Ut, pagelocked, mainGPU,
	                           numGPUs, gpuArray );
}

const tval3_info<float> tval3_gpu_3d(mat_host<float> &U,
		const geometry_host &A_geo, sparse_mat_host<float> &A_mat,
		const mat_host<float> &b, const tval3_options<float> &opts,
		const mat_host<float> &Ut, bool pagelocked, int mainGPU, int numGPUs, 
		int *gpuArray ) {

	return tval3_gpu_3d<float>(U, A_geo, A_mat, b, opts, Ut, pagelocked,
			mainGPU, numGPUs, gpuArray );
}

const tval3_info<double> tval3_gpu_3d(mat_host<double> &U,
		const sparse_mat_host<double> &A, const mat_host<double> &b,
		const tval3_options<double> &opts, const mat_host<double> &Ut,
		bool pagelocked, int mainGPU, int numGPUs, int *gpuArray ) {

	return tval3_gpu_3d<double>(U, A, b, opts, Ut, pagelocked, mainGPU,
			numGPUs, gpuArray );
}

const tval3_info<double> tval3_gpu_3d(mat_host<double> &U,
		const mat_host<double> &A, const mat_host<double> &b,
		const tval3_options<double> &opts, const mat_host<double> &Ut,
		bool pagelocked, int mainGPU, int numGPUs, int *gpuArray ) {

	return tval3_gpu_3d<double>(U, A, b, opts, Ut, pagelocked, mainGPU,
			numGPUs, gpuArray );
}

const tval3_info<double> tval3_gpu_3d(mat_host<double> &U,
		const geometry_host &A, const mat_host<double> &b,
		const tval3_options<double> &opts, const mat_host<double> &Ut,
		bool pagelocked, int mainGPU, int numGPUs, int *gpuArray ) {

	return tval3_gpu_3d<double>(U, A, b, opts, Ut, pagelocked, mainGPU,
			numGPUs, gpuArray );
}

const tval3_info<double> tval3_gpu_3d(mat_host<double> &U,
		const geometry_host &A_geo, sparse_mat_host<double> &A_mat,
		const mat_host<double> &b, const tval3_options<double> &opts,
		const mat_host<double> &Ut, bool pagelocked, int mainGPU, int numGPUs,
		int *gpuArray ) {

	return tval3_gpu_3d<double>(U, A_geo, A_mat, b, opts, Ut, pagelocked,
	                            mainGPU, numGPUs, gpuArray );
}
