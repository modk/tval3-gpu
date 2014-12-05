#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include "handle_error.h"
#include "container_device.h"
#include "tval3_gpu.h"
#include "mat_vec_mul.h"

// dynamically allocated shared memory array
extern __shared__ float buffer[];

// texture references
texture<float, 2> texRef;
texture<float, 2> texRefX;
texture<float, 2> texRefY;

cudaChannelFormatDesc channelDesc;

cudaStream_t stream1, stream2, stream3;

int majorRevision;

cublasHandle_t cb_handle;
cusparseHandle_t cs_handle;
cusparseMatDescr_t descrA;

// this struct holds various values, accessed by the functions get_g(), 
// update_g(), update_W() and update_mlp()
// the member names correspond to the variable names in the MATLAB-version
struct tval3_data_gpu { // parameters and intermediate results
	// dimensions
	int p; // #rows of reconstruction "volume"
	int q; // #cols of reconstruction "volume"
	int n; // number of pixels (n = p*q)
	int m; // number of measurements

	// penalty parameters
	float mu;
	float beta;
	float muDbeta; // mu/beta

	// multiplier
	mat_device delta;
	mat_device sigmaX;
	mat_device sigmaY;

	// lagrangian function and sub terms
	float f;
	float lam1;
	float lam2;
	float lam3;
	float lam4;
	float lam5;

	// other intermediate results
	mat_device Up;
	mat_device dU; // U-Up, after steepest descent
	mat_device uup; // U-Up, after "backtracking"
	mat_device Ux; // gradients in x-direction
	mat_device Uxp;
	mat_device dUx;
	mat_device Uy; // gradients in y-direction
	mat_device Uyp;
	mat_device dUy;
	mat_device Wx;
	mat_device Wy;
	mat_device Atb; // A'*b
	mat_device Au; // A*u
	mat_device Aup;
	mat_device dAu;
	mat_device d; // gradient of objective function (2.28)
	mat_device g; // sub term of (2.28)
	mat_device gp;
	mat_device dg;
	mat_device g2; // sub term of (2.28)
	mat_device g2p;
	mat_device dg2;

	// buffers for reductions
	mat_device gpu_buff1; // length: |U|
	mat_device gpu_buff2; // length: |U|
	mat_device gpu_buff3; // length: |b|
	mat_device gpu_buff4; // length: |b|
	mat_host host_buff1; // length: |U|
	mat_host host_buff2; // length: |U|
	mat_host host_buff3; // length: |b|
	mat_host host_buff4; // length: |b|

	tval3_data_gpu(const mat_host &U, const mat_host &b, float mu0,
	               float beta0, bool pagelocked) :
		p(U.rows), q(U.cols), n(U.len), m(b.len), mu(mu0), beta(beta0),
		delta(b.len), sigmaX(U.rows, U.cols), sigmaY(U.rows, U.cols),
		Up(U.rows, U.cols, false),
		dU(U.rows, U.cols, false), 
		uup(U.len), Ux(U.rows, U.cols, false),
		Uxp(U.rows, U.cols, false), 
		dUx(U.rows, U.cols, false), 
		Uy(U.rows, U.cols, false), 
		Uyp(U.rows, U.cols, false),
		dUy(U.rows, U.cols, false), 
		Wx(U.rows, U.cols, false), 
		Wy( U.rows, U.cols, false), 
		Atb(U.len, false), 
		Au(b.len, false), 
		Aup(b.len, false), 
		dAu(b.len, false), 
		d(U.rows, U.cols, false), 
		g(U.len, false), 
		gp(U.len, false),
		dg(U.len, false), 
		g2(U.len, false), 
		g2p(U.len, false), 
		dg2( U.len, false), gpu_buff1(U.len, false), gpu_buff2(U.len, false),
		host_buff1(U.len, false, pagelocked), host_buff2(U.len, false, pagelocked),
		gpu_buff3(b.len, false), gpu_buff4(b.len, false),
		host_buff3(b.len, false, pagelocked), host_buff4(b.len, false, pagelocked)
	{ }
};

// result = alpha*x + y + z
// z = NULL allowed
__global__ void vec_add_kernel(int N, float *result, const float alpha,
                               const float *x, const float *y, const float *z) {
	int index = (threadIdx.x + blockIdx.x * blockDim.x);

	float val;

	while (index < N) {
		val = alpha * x[index] + y[index];

		if (z != NULL)
			val += z[index];
		result[index] = val;
		index += gridDim.x * blockDim.x;
	}
}

// z <- alpha*x + y
// N: number of elements
inline void vec_add_gpu(const int N, const int stride, const float *y,
                        const float alpha, const float *x, float *z,
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

	vec_add_kernel << < blocks, threads, 0, stream >> >
	        (N, z, alpha, x, y, NULL);
}

// Scales the measurement vector. Returns the scale factor.
float scaleb_gpu(const mat_host &b, mat_device &b_gpu) {
	float scl = 1;

	if (b.len > 0) {
		float threshold1 = 0.5;
		float threshold2 = 1.5;
		scl = 1;
		float val;
		float val_abs;
		float bmin = b[0];
		float bmax = bmin;
		float bmin_abs = abs(bmin);
		float bmax_abs = bmin_abs;
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
		float b_dif = abs(bmax - bmin);

		if (b_dif < threshold1) {
			scl = threshold1 / b_dif;
			HANDLE_ERROR(cublasSscal(cb_handle, b_gpu.len, &scl,
			                         b_gpu.data_dev_ptr(), 1));
		} else if (b_dif > threshold2) {
			scl = threshold2 / b_dif;
			HANDLE_ERROR(cublasSscal(cb_handle, b_gpu.len, &scl,
			                         b_gpu.data_dev_ptr(), 1));
		}
	}
	return scl;
}

// matrices stored in row major format
__global__ void D_kernel_no_tex(const float *U, float *Ux, float *Uy, int w,
                                int h) {

	int x, y, lin_index;

	float tmp;

	y = blockIdx.y * blockDim.y + threadIdx.y;

	while (y < h) {
		x = blockIdx.x * blockDim.x + threadIdx.x;

		while (x < w) {
			lin_index = y * w + x;

			tmp = U[lin_index];

			if (x < w - 1)
				Ux[lin_index] = U[lin_index + 1] - tmp;
			else
				Ux[lin_index] = U[lin_index - x] - tmp;

			if (y < h - 1)
				Uy[lin_index] = U[lin_index + w] - tmp;
			else
				Uy[lin_index] = U[x] - tmp;

			x += gridDim.x * blockDim.x;
		}
		y += gridDim.y * blockDim.y;
	}
}

__global__ void D_kernel_tex(const float *U, float *Ux, float *Uy, int w,
                             int h) {

	int x, y, lin_index;

	float tmp;

	y = blockIdx.y * blockDim.y + threadIdx.y;

	while (y < h) {
		x = blockIdx.x * blockDim.x + threadIdx.x;

		while (x < w) {
			lin_index = y * w + x;

			tmp = tex2D(texRef, x, y);

			if (x < w - 1) {

				Ux[lin_index] = tex2D(texRef, x + 1, y) - tmp;
			} else {
				Ux[lin_index] = tex2D(texRef, 0, y) - tmp;
			}

			if (y < h - 1) {
				Uy[lin_index] = tex2D(texRef, x, y + 1) - tmp;
			} else {
				Uy[lin_index] = tex2D(texRef, x, 0) - tmp;
			}

			x += gridDim.x * blockDim.x;
		}
		y += gridDim.y * blockDim.y;
	}
}

// matrices stored in column major format
void D_gpu(mat_device &Ux, mat_device &Uy, const mat_device &U) {
	dim3 threads, blocks;

	if (majorRevision >= 2) {
		threads.x = 64;
		threads.y = 4;
		blocks.x = max((int)round((double)U.rows / threads.x / 4), 1);
		blocks.y = max((int)round((double)U.cols / threads.y), 1);
		D_kernel_no_tex << < blocks, threads >> >
		        (U.data_dev_ptr(), Uy.data_dev_ptr(), Ux.data_dev_ptr(),
		         U.rows,
		         U.cols);
	} else {
		threads.x = 32;
		threads.y = 4;
		blocks.x = max((int)round((double)U.rows / threads.x / 2), 1);
		blocks.y = max((int)round((double)U.cols / threads.y), 1);

		HANDLE_ERROR(cudaBindTexture2D(NULL, &texRef, U.data_dev_ptr(),
		                               &channelDesc,
		                               U.rows, U.cols, U.rows *
		                               sizeof(float)));

		D_kernel_tex << < blocks, threads >> >
		        (U.data_dev_ptr(), Uy.data_dev_ptr(), Ux.data_dev_ptr(),
		         U.rows,
		         U.cols);

		HANDLE_ERROR(cudaUnbindTexture(texRef));
	}
}

__global__ void Dt_kernel_no_tex(const float *X, const float *Y, float *res,
                                 int w, int h) {

	int x, y, lin_index;
	float xp, yp;

	y = blockIdx.y * blockDim.y + threadIdx.y;

	while (y < h) {

		x = blockIdx.x * blockDim.x + threadIdx.x;

		while (x < w) {

			lin_index = y * w + x;

			xp = (x == 0) ? X[lin_index + w - 1] : X[lin_index - 1];
			yp =
			        (y == 0) ? Y[lin_index +
			                     (h - 1) * w] : Y[lin_index - w];

			res[lin_index] = xp - X[lin_index] + yp - Y[lin_index];

			x += blockDim.x * gridDim.x;
		}
		y += blockDim.y * gridDim.y;
	}
}

// texture mem
__global__ void Dt_kernel_tex(const float *X, const float *Y, float *res, int w,
                              int h) {

	int x, y, lin_index;
	float xp, yp;

	y = blockIdx.y * blockDim.y + threadIdx.y;

	while (y < h) {

		x = blockIdx.x * blockDim.x + threadIdx.x;

		while (x < w) {

			lin_index = y * w + x;

			xp = (x == 0) ? tex2D(texRefX, w - 1, y) : tex2D(
			        texRefX, x - 1, y);
			yp = (y == 0) ? tex2D(texRefY, x, h - 1) : tex2D(
			        texRefY, x, y - 1);

			res[lin_index] = xp - tex2D(texRefX, x, y) + yp - tex2D(
			        texRefY, x, y);

			x += blockDim.x * gridDim.x;
		}
		y += blockDim.y * gridDim.y;
	}
}

void Dt_gpu(mat_device &res, const mat_device &X, const mat_device &Y,
            cudaStream_t stream = 0) {
	dim3 threads, blocks;

	if (majorRevision >= 2) {
		threads.x = 128;
		threads.y = 2;
		blocks.x = max((int)round((double)X.rows / threads.x / 4), 1);
		blocks.y = max((int)round((double)X.cols / threads.y / 2), 1);
		Dt_kernel_no_tex << < blocks, threads, 0, stream >> >
		        (Y.data_dev_ptr(), X.data_dev_ptr(), res.data_dev_ptr(),
		         X.rows,
		         X.cols);
	} else {
		threads.x = 32;
		threads.y = 4;
		blocks.x = max((int)round((double)X.rows / threads.x / 2), 1);
		blocks.y = max((int)round((double)X.cols / threads.y), 1);

		HANDLE_ERROR(cudaBindTexture2D(NULL, &texRefX, X.data_dev_ptr(),
		                               &channelDesc,
		                               X.rows, X.cols, X.rows *
		                               sizeof(float)));
		HANDLE_ERROR(cudaBindTexture2D(NULL, &texRefY, Y.data_dev_ptr(),
		                               &channelDesc,
		                               Y.rows, Y.cols, Y.rows *
		                               sizeof(float)));

		Dt_kernel_tex << < blocks, threads, 0, stream >> >
		        (Y.data_dev_ptr(), X.data_dev_ptr(), res.data_dev_ptr(),
		         X.rows,
		         X.cols);

		HANDLE_ERROR(cudaUnbindTexture(texRefX));
		HANDLE_ERROR(cudaUnbindTexture(texRefY));
	}
}

__global__ void lam2_4_kernel(int N, const float *Vx, const float *Vy,
                              const float *sigmaX, const float *sigmaY,
                              float *tmp_lam2, float *tmp_lam4) {

	// pointers to dynamically allocated shared memory
	float *lam2_buffer = buffer;
	float *lam4_buffer = buffer + blockDim.x;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float Vxi, Vyi, sum_lam2 = 0, sum_lam4 = 0;

	while (index < N) {
		Vxi = Vx[index];
		Vyi = Vy[index];
		sum_lam2 += Vxi * Vxi + Vyi * Vyi;
		sum_lam4 += Vxi * sigmaX[index] + Vyi * sigmaY[index];

		index += gridDim.x * blockDim.x;
	}

	lam2_buffer[threadIdx.x] = sum_lam2;
	lam4_buffer[threadIdx.x] = sum_lam4;

	__syncthreads();

	// reduction

	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			lam2_buffer[threadIdx.x] +=
			        lam2_buffer [threadIdx.x + i];
			lam4_buffer[threadIdx.x] +=
			        lam4_buffer [threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		tmp_lam2[blockIdx.x] = lam2_buffer[0];
		tmp_lam4[blockIdx.x] = lam4_buffer[0];
	}
}

__global__ void lam3_5_kernel(int N, const float *Au, const float *b,
                              const float *delta, float *tmp_lam3,
                              float *tmp_lam5) {

	// pointers to dynamically allocated shared memory
	float *lam3_buffer = buffer;
	float *lam5_buffer = buffer + blockDim.x;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float Aub, sum_lam3 = 0, sum_lam5 = 0;

	while (index < N) {
		Aub = Au[index] - b[index];
		sum_lam3 += Aub * Aub;
		sum_lam5 += delta[index] * Aub;

		index += gridDim.x * blockDim.x;
	}

	lam3_buffer[threadIdx.x] = sum_lam3;
	lam5_buffer[threadIdx.x] = sum_lam5;

	__syncthreads();

	// reduction

	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			lam3_buffer[threadIdx.x] +=
			        lam3_buffer [threadIdx.x + i];
			lam5_buffer[threadIdx.x] +=
			        lam5_buffer [threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		tmp_lam3[blockIdx.x] = lam3_buffer[0];
		tmp_lam5[blockIdx.x] = lam5_buffer[0];
	}
}

template<class matrix_type>
void get_g_gpu(tval3_data_gpu &data, const matrix_type &A, const mat_device &U,
               const mat_device &b) {
	// set block- and grid dimensions for lam2_4_kernel and lam3_5_kernel
	int threads = 128;
	int blocks = (majorRevision >= 2) ? 96 : 90;
	int blocks_lam2_4 = min(blocks, (data.n + threads - 1) / threads);
	int blocks_lam3_5 = min(blocks, (data.m + threads - 1) / threads);

	// Au = A*U
	mat_vec_mul(CUBLAS_OP_N, A, U.data_dev_ptr(),
	            data.Au.data_dev_ptr(), stream1);

	// g = A'*Au
	mat_vec_mul(CUBLAS_OP_T, A, data.Au.data_dev_ptr(),
	            data.g.data_dev_ptr(), stream1);

	// update g2, lam2, lam4
	mat_device Vx(data.p, data.q, false);
	mat_device Vy(data.p, data.q, false);
	vec_add_gpu(data.n, 1,
	            data.Ux.data_dev_ptr(), (float)-1,
	            data.Wx.data_dev_ptr(), Vx.data_dev_ptr(), stream2);                                               // Vx = Ux - Wx
	vec_add_gpu(data.n, 1,
	            data.Uy.data_dev_ptr(), (float)-1,
	            data.Wy.data_dev_ptr(), Vy.data_dev_ptr(), stream3);                                               // Vy = Uy - Wy

	lam3_5_kernel << < blocks_lam3_5, threads, threads * 2 * sizeof(float),
	        stream2 >> >
	        (data.m, data.Au.data_dev_ptr(), b.data_dev_ptr(),
	         data.delta.data_dev_ptr(),
	         data
	         .gpu_buff3.data_dev_ptr(), data.gpu_buff4.data_dev_ptr());

	lam2_4_kernel << < blocks_lam2_4, threads, threads * 2 * sizeof(float),
	        stream3 >> >
	        (data.n, Vx.data_dev_ptr(), Vy.data_dev_ptr(),
	         data.sigmaX.data_dev_ptr(),
	         data
	         .sigmaY.data_dev_ptr(), data.gpu_buff1.data_dev_ptr(),
	         data.gpu_buff2.data_dev_ptr());

	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff3.data(),
	                             data.gpu_buff3.data_dev_ptr(),
	                             blocks_lam3_5 * sizeof(float),
	                             cudaMemcpyDeviceToHost, stream2));
	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff4.data(),
	                             data.gpu_buff4.data_dev_ptr(),
	                             blocks_lam3_5 * sizeof(float),
	                             cudaMemcpyDeviceToHost, stream2));

	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff1.data(),
	                             data.gpu_buff1.data_dev_ptr(),
	                             blocks_lam2_4 * sizeof(float),
	                             cudaMemcpyDeviceToHost, stream3));
	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff2.data(),
	                             data.gpu_buff2.data_dev_ptr(),
	                             blocks_lam2_4 * sizeof(float),
	                             cudaMemcpyDeviceToHost, stream3));


	Dt_gpu(data.g2, Vx, Vy, stream2);

	HANDLE_ERROR(cudaStreamSynchronize(stream2));

	float sum3 = 0, sum4 = 0;

	for (int i = 0; i < blocks_lam3_5; i++) {
		sum3 += data.host_buff3[i];
		sum4 += data.host_buff4[i];
	}

	data.lam3 = sum3;
	data.lam5 = sum4;

	HANDLE_ERROR(cudaStreamSynchronize(stream3));

	float sum1 = 0, sum2 = 0;

	for (int i = 0; i < blocks_lam2_4; i++) {
		sum1 += data.host_buff1[i];
		sum2 += data.host_buff2[i];
	}

	data.lam2 = sum1;
	data.lam4 = sum2;

	HANDLE_ERROR(cudaStreamSynchronize(stream1));

	// g = g - Atb
	float val_alpha = -1;
	HANDLE_ERROR(cublasSaxpy(cb_handle, data.n, &val_alpha,
	                         data.Atb.data_dev_ptr(), 1,
	                         data.g.data_dev_ptr(), 1));

	data.f =
	        (data.lam1 + data.beta / 2 * data.lam2 + data.mu / 2 *
	         data.lam3) - data.lam4 - data.lam5;
}

// slower version (in micro benchmark faster!)
/*
   template<class matrix_type>
   void get_g_gpu(tval3_data_gpu &data, const matrix_type &A, const mat_device &U,
                const mat_device &b, bool rec_desc_time=false) {

        // set block- and grid dimensions for lam2_4_kernel and lam3_5_kernel
        int threads = 128;
        int blocks = (majorRevision >= 2) ? 96 : 90;
        int blocks_lam2_4 = min(blocks, (data.n + threads - 1) / threads);
        int blocks_lam3_5 = min(blocks, (data.m + threads - 1) / threads);

        mat_device Vx(data.p, data.q, false);
        mat_device Vy(data.p, data.q, false);

        // Au = A*U
        mat_vec_mul(CUBLAS_OP_N, A, U.data_dev_ptr(), data.Au.data_dev_ptr(), stream1);

        // update g2, lam2, lam4
        vec_add_gpu(data.n, 1, data.Ux.data_dev_ptr(), (float)-1, data.Wx.data_dev_ptr(), Vx.data_dev_ptr(), stream2); // Vx = Ux - Wx
        vec_add_gpu(data.n, 1, data.Uy.data_dev_ptr(), (float)-1, data.Wy.data_dev_ptr(), Vy.data_dev_ptr(), stream3); // Vy = Uy - Wy

        // g = A'*Au
        mat_vec_mul(CUBLAS_OP_T, A, data.Au.data_dev_ptr(), data.g.data_dev_ptr(), stream1);

        lam3_5_kernel<<<blocks_lam3_5, threads, threads * 2 * sizeof(float), stream2>>>(data.m, data.Au.data_dev_ptr(), b.data_dev_ptr(), data.delta.data_dev_ptr(),
                        data.gpu_buff3.data_dev_ptr(), data.gpu_buff4.data_dev_ptr());

        lam2_4_kernel<<<blocks_lam2_4, threads, threads * 2 * sizeof(float), stream3>>>(data.n, Vx.data_dev_ptr(), Vy.data_dev_ptr(), data.sigmaX.data_dev_ptr(),
                        data.sigmaY.data_dev_ptr(), data.gpu_buff1.data_dev_ptr(), data.gpu_buff2.data_dev_ptr());

        HANDLE_ERROR(cudaMemcpyAsync(data.host_buff3.data(), data.gpu_buff3.data_dev_ptr(), blocks_lam3_5*sizeof(float), cudaMemcpyDeviceToHost, stream2));
        HANDLE_ERROR(cudaMemcpyAsync(data.host_buff4.data(), data.gpu_buff4.data_dev_ptr(), blocks_lam3_5*sizeof(float), cudaMemcpyDeviceToHost, stream2));

        HANDLE_ERROR(cudaMemcpyAsync(data.host_buff1.data(), data.gpu_buff1.data_dev_ptr(), blocks_lam2_4*sizeof(float), cudaMemcpyDeviceToHost, stream3));
        HANDLE_ERROR(cudaMemcpyAsync(data.host_buff2.data(), data.gpu_buff2.data_dev_ptr(), blocks_lam2_4*sizeof(float), cudaMemcpyDeviceToHost, stream3));

        Dt_gpu(data.g2, Vx, Vy, stream2);

        HANDLE_ERROR(cudaStreamSynchronize(stream1));

        // g = g - Atb
        float val_alpha = -1;
        HANDLE_ERROR(cublasSaxpy(cb_handle, data.n, &val_alpha, data.Atb.data_dev_ptr(), 1, data.g.data_dev_ptr(), 1));

        HANDLE_ERROR(cudaStreamSynchronize(stream2));

        float sum3=0, sum4=0;

        for(int i=0; i < blocks_lam3_5; i++) {
                sum3 += data.host_buff3[i];
                sum4 += data.host_buff4[i];
        }

        data.lam3 = sum3;
        data.lam5 = sum4;

        HANDLE_ERROR(cudaStreamSynchronize(stream3));

        float sum1=0, sum2=0;

        for(int i=0; i < blocks_lam2_4; i++) {
                sum1 += data.host_buff1[i];
                sum2 += data.host_buff2[i];
        }

        data.lam2 = sum1;
        data.lam4 = sum2;

        data.f = (data.lam1 + data.beta/2*data.lam2 + data.mu/2*data.lam3) - data.lam4 - data.lam5;
   }*/

__global__ void get_tau_kernel(int N, const float *g, const float *gp,
                               const float *g2, const float *g2p,
                               const float *uup,
                               float muDbeta, float *dg, float *dg2,
                               float *tmp_ss, float *tmp_sy) {

	// pointers to dynamically allocated shared memory
	float *ss = buffer;
	float *sy = buffer + blockDim.x;

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	float uup_i, dg_i, dg2_i, sum_ss = 0, sum_sy = 0;

	while (index < N) {
		dg_i = g[index] - gp[index];
		dg2_i = g2[index] - g2p[index];

		uup_i = uup[index];

		sum_ss += uup_i * uup_i;
		sum_sy += uup_i * (dg2_i + muDbeta * dg_i);

		dg[index] = dg_i;
		dg2[index] = dg2_i;

		index += gridDim.x * blockDim.x;
	}

	ss[threadIdx.x] = sum_ss;
	sy[threadIdx.x] = sum_sy;

	__syncthreads();

	// reduction

	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			ss[threadIdx.x] += ss [threadIdx.x + i];
			sy[threadIdx.x] += sy [threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		tmp_ss[blockIdx.x] = ss[0];
		tmp_sy[blockIdx.x] = sy[0];
	}
}

template<class matrix_type>
float get_tau_gpu(tval3_data_gpu &data, const matrix_type &A, bool fst_iter) {
	float tau;

	// calculate tau
	if (fst_iter) {
		mat_device dx(data.p, data.q, false);
		mat_device dy(data.p, data.q, false);
		D_gpu(dx, dy, data.d);

		float result1, result2;
		HANDLE_ERROR(cublasSdot(cb_handle, data.n, dx.data_dev_ptr(), 1,
		                        dx.data_dev_ptr(), 1, &result1));
		HANDLE_ERROR(cublasSdot(cb_handle, data.n, dy.data_dev_ptr(), 1,
		                        dy.data_dev_ptr(), 1, &result2));
		float dDd = result1 + result2;

		float dd;
		HANDLE_ERROR(cublasSdot(cb_handle, data.n,
		                        data.d.data_dev_ptr(), 1,
		                        data.d.data_dev_ptr(), 1, &dd));

		mat_device Ad(data.m, false);

		// Ad=A*d
		mat_vec_mul(CUBLAS_OP_N, A,
		            data.d.data_dev_ptr(), Ad.data_dev_ptr());

		float Add;
		HANDLE_ERROR(cublasSdot(cb_handle, data.m, Ad.data_dev_ptr(), 1,
		                        Ad.data_dev_ptr(), 1, &Add));

		tau = abs(dd / (dDd + data.muDbeta * Add));
	} else {

		// set block- and grid dimensions for get_tau_kernel
		int threads = 128;
		int blocks = (majorRevision >= 2) ? 64 : 90;
		blocks = min(blocks, (data.n + threads - 1) / threads);

		get_tau_kernel << < blocks, threads, 2 * threads *
		        sizeof(float) >> >
		        (data.n, data.g.data_dev_ptr(), data.gp.data_dev_ptr(),
		         data
		         .g2.data_dev_ptr(), data.g2p.data_dev_ptr(),
		         data.uup.data_dev_ptr(), data.muDbeta,
		         data.dg.data_dev_ptr(),
		         data
		         .dg2.data_dev_ptr(), data.gpu_buff1.data_dev_ptr(),
		         data.gpu_buff2.data_dev_ptr());

		HANDLE_ERROR(cudaMemcpy(data.host_buff1.data(),
		                        data.gpu_buff1.data_dev_ptr(), blocks *
		                        sizeof(float),
		                        cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(data.host_buff2.data(),
		                        data.gpu_buff2.data_dev_ptr(), blocks *
		                        sizeof(float),
		                        cudaMemcpyDeviceToHost));

		float ss = 0, sy = 0;

		for (int i = 0; i < blocks; i++) {
			ss += data.host_buff1[i];
			sy += data.host_buff2[i];
		}

		tau = abs(ss / sy);
	}
	return tau;
}


__global__ void nonneg_kernel(int N, float *vec) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	float val;
	while (index < N) {
		val = vec[index];
		vec[index] = (val < 0) ? 0 : val;

		index += gridDim.x * blockDim.x;
	}
}

template<class matrix_type>
void descend_gpu(mat_device &U, tval3_data_gpu &data, const matrix_type &A,
                 const mat_device &b, float tau,
                 bool nonneg) {
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

	float val_alpha = -1 * tau;
	HANDLE_ERROR(cublasSaxpy(cb_handle, data.n, &val_alpha,
	                         data.d.data_dev_ptr(), 1, U.data_dev_ptr(),
	                         1));                                                                           // U = U - tau*d

	if (nonneg)
		nonneg_kernel << < blocks, threads >> >
		        (data.n, U.data_dev_ptr());

	D_gpu(data.Ux, data.Uy, U);

	get_g_gpu(data, A, U, b);
}

template<class matrix_type>
void update_g_gpu(tval3_data_gpu &data, const float alpha, const matrix_type &A,
                  mat_device &U, const mat_device &b) {

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

	// update lam2, lam4
	mat_device Vx(data.p, data.q, false);
	mat_device Vy(data.p, data.q, false);
	vec_add_gpu(data.n, 1,
	            data.Ux.data_dev_ptr(), (float)-1,
	            data.Wx.data_dev_ptr(), Vx.data_dev_ptr(), stream2);                                               // Vx = Ux - Wx
	vec_add_gpu(data.n, 1,
	            data.Uy.data_dev_ptr(), (float)-1,
	            data.Wy.data_dev_ptr(), Vy.data_dev_ptr(), stream3);                                               // Vy = Uy - Wy

	HANDLE_ERROR(cudaStreamSynchronize(stream1));

	lam3_5_kernel << < blocks_lam3_5, threads, threads * 2 * sizeof(float),
	        stream2 >> >
	        (data.m, data.Au.data_dev_ptr(), b.data_dev_ptr(),
	         data.delta.data_dev_ptr(),
	         data
	         .gpu_buff3.data_dev_ptr(), data.gpu_buff4.data_dev_ptr());

	lam2_4_kernel << < blocks_lam2_4, threads, threads * 2 * sizeof(float),
	        stream3 >> >
	        (data.n, Vx.data_dev_ptr(), Vy.data_dev_ptr(),
	         data.sigmaX.data_dev_ptr(),
	         data
	         .sigmaY.data_dev_ptr(), data.gpu_buff1.data_dev_ptr(),
	         data.gpu_buff2.data_dev_ptr());

	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff3.data(),
	                             data.gpu_buff3.data_dev_ptr(),
	                             blocks_lam3_5 * sizeof(float),
	                             cudaMemcpyDeviceToHost, stream2));
	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff4.data(),
	                             data.gpu_buff4.data_dev_ptr(),
	                             blocks_lam3_5 * sizeof(float),
	                             cudaMemcpyDeviceToHost, stream2));

	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff1.data(),
	                             data.gpu_buff1.data_dev_ptr(),
	                             blocks_lam2_4 * sizeof(float),
	                             cudaMemcpyDeviceToHost, stream3));
	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff2.data(),
	                             data.gpu_buff2.data_dev_ptr(),
	                             blocks_lam2_4 * sizeof(float),
	                             cudaMemcpyDeviceToHost, stream3));

	HANDLE_ERROR(cudaStreamSynchronize(stream2));

	float sum3 = 0, sum4 = 0;

	for (int i = 0; i < blocks_lam3_5; i++) {
		sum3 += data.host_buff3[i];
		sum4 += data.host_buff4[i];
	}

	data.lam3 = sum3;
	data.lam5 = sum4;

	HANDLE_ERROR(cudaStreamSynchronize(stream3));

	float sum1 = 0, sum2 = 0;

	for (int i = 0; i < blocks_lam2_4; i++) {
		sum1 += data.host_buff1[i];
		sum2 += data.host_buff2[i];
	}

	data.lam2 = sum1;
	data.lam4 = sum2;


	data.f = data.lam1 + data.beta / 2 * data.lam2 + data.mu / 2 *
	         data.lam3 - data.lam4 - data.lam5;
}

template<class matrix_type>
void min_u_gpu(tval3_data_gpu &data, mat_device &U, float &gam,
               const matrix_type &A, const mat_device &b,
               const tval3_options &opts, float C, bool fst_iter) {
	float tau, alpha, c_armin;
	tau = get_tau_gpu(data, A, fst_iter);

	// keep previous values
	data.Up = U; data.gp = data.g; data.g2p = data.g2;
	data.Aup = data.Au; data.Uxp = data.Ux; data.Uyp = data.Uy;

	// one step steepest descend
	descend_gpu(U, data, A, b, tau, opts.nonneg);

	// NMLA
	alpha = 1;
	vec_add_gpu(data.n, 1, U.data_dev_ptr(), -1,
	            data.Up.data_dev_ptr(), data.dU.data_dev_ptr());                                  // dU = U - Up
	HANDLE_ERROR(cublasSdot(cb_handle, data.n, data.d.data_dev_ptr(), 1,
	                        data.d.data_dev_ptr(), 1, &c_armin));                                              // c_armin = d'*d
	c_armin *= tau * opts.c * data.beta;

	if (abs(data.f) > abs(C - alpha * c_armin)) {  // Arminjo condition

		vec_add_gpu(data.n, 1,
		            data.g.data_dev_ptr(), -1,
		            data.gp.data_dev_ptr(), data.dg.data_dev_ptr());                                       // dg=g-gp
		vec_add_gpu(data.n, 1,
		            data.g2.data_dev_ptr(), -1,
		            data.g2p.data_dev_ptr(), data.dg2.data_dev_ptr());                                        // dg2=g2-g2p
		vec_add_gpu(data.m, 1,
		            data.Au.data_dev_ptr(), -1,
		            data.Aup.data_dev_ptr(), data.dAu.data_dev_ptr());                                        // dAu=Au-Aup
		vec_add_gpu(data.n, 1,
		            data.Ux.data_dev_ptr(), -1,
		            data.Uxp.data_dev_ptr(), data.dUx.data_dev_ptr());                                        // dUx=Ux-Uxp
		vec_add_gpu(data.n, 1,
		            data.Uy.data_dev_ptr(), -1,
		            data.Uyp.data_dev_ptr(), data.dUy.data_dev_ptr());                                        // Uy = Uy-Uyp

		int cnt = 0;
		while (abs(data.f) > abs(C - alpha * c_armin)) { // Arminjo condition
			if (cnt == 5) { // "backtracking" not successful
				gam *= opts.rate_gam;
				tau = get_tau_gpu(data, A, true);
				U = data.Up;
				descend_gpu(U, data, A, b, tau, opts.nonneg);
				break;
			}
			alpha *= opts.gamma;

			update_g_gpu(data, alpha, A, U, b);

			cnt++;
		}
	}
}

void get_gradient_gpu(tval3_data_gpu &data, const mat_device &DtsAtd) {

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

	// d = g2 + muDbeta*g + DtsAtd   (DtsAtd has opposite sign, compared to the MATLAB-version!)
	vec_add_kernel << < blocks, threads >> >
	        (data.n, data.d.data_dev_ptr(), data.muDbeta,
	         data.g.data_dev_ptr(),
	         data.g2.data_dev_ptr(), DtsAtd.data_dev_ptr());
}

__global__ void shrinkage_kernel(int N, const float *Ux, const float *Uy,
                                 const float *sigmaX, const float *sigmaY,
                                 float beta, float *Wx, float *Wy, float *tmp) {

	float Uxbar, Uybar, temp_wx, temp_wy;

	// pointer to dynamically allocated shared memory
	float *buff = buffer;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0;

	while (index < N) {
		Uxbar = Ux[index] - sigmaX[index] / beta;
		Uybar = Uy[index] - sigmaY[index] / beta;
		temp_wx = abs(Uxbar) - 1 / beta;
		temp_wy = abs(Uybar) - 1 / beta;
		temp_wx = (temp_wx >= 0) ? temp_wx : 0;
		temp_wy = (temp_wy >= 0) ? temp_wy : 0;
		Wx[index] = (Uxbar >= 0) ? temp_wx : -temp_wx;
		Wy[index] = (Uybar >= 0) ? temp_wy : -temp_wy;

		sum += temp_wx + temp_wy;

		index += gridDim.x * blockDim.x;
	}

	buff[threadIdx.x] = sum;

	__syncthreads();

	// reduction

	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			buff[threadIdx.x] += buff [threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		tmp[blockIdx.x] = buff[0];
	}
}

void shrinkage_gpu(tval3_data_gpu &data) {
	int threads, blocks;

	if (majorRevision >= 2) {
		threads = 128;
		blocks = 64;
	} else {
		threads = 128;
		blocks = 90;
	}
	blocks = min(blocks, (data.n + threads - 1) / threads);

	shrinkage_kernel << < blocks, threads, threads * sizeof(float) >> >
	        (data.n, data.Ux.data_dev_ptr(), data.Uy.data_dev_ptr(),
	         data
	         .sigmaX.data_dev_ptr(), data.sigmaY.data_dev_ptr(), data.beta,
	         data.Wx.data_dev_ptr(),
	         data
	         .Wy.data_dev_ptr(), data.gpu_buff1.data_dev_ptr());

	HANDLE_ERROR(cudaMemcpy(data.host_buff1.data(),
	                        data.gpu_buff1.data_dev_ptr(), blocks *
	                        sizeof(float),
	                        cudaMemcpyDeviceToHost));

	float sum = 0;
	for (int i = 0; i < blocks; i++)
		sum += data.host_buff1[i];

	data.lam1 = sum;
}

void update_W_gpu(tval3_data_gpu &data) {

	// set block- and grid dimensions for lam2_4_kernel
	int threads = 128;
	int blocks = (majorRevision >= 2) ? 96 : 90;
	blocks = min(blocks, (data.n + threads - 1) / threads);

	data.f -= (data.lam1 + data.beta / 2 * data.lam2 - data.lam4);

	shrinkage_gpu(data);

	// update g2, lam2, lam4
	mat_device Vx(data.p, data.q, false);
	mat_device Vy(data.p, data.q, false);
	// Vx = Ux - Wx
	vec_add_gpu(data.n, 1,
	            data.Ux.data_dev_ptr(), (float)-1,
	            data.Wx.data_dev_ptr(), Vx.data_dev_ptr(), stream1);
	// Vy = Uy - Wy
	vec_add_gpu(data.n, 1,
	            data.Uy.data_dev_ptr(), (float)-1,
	            data.Wy.data_dev_ptr(), Vy.data_dev_ptr(), stream2);

	HANDLE_ERROR(cudaStreamSynchronize(stream1));
	HANDLE_ERROR(cudaStreamSynchronize(stream2));

	lam2_4_kernel << < blocks, threads, threads * 2 * sizeof(float),
	        stream1 >> >
	        (data.n, Vx.data_dev_ptr(), Vy.data_dev_ptr(),
	         data.sigmaX.data_dev_ptr(),
	         data
	         .sigmaY.data_dev_ptr(), data.gpu_buff1.data_dev_ptr(),
	         data.gpu_buff2.data_dev_ptr());

	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff1.data(),
	                             data.gpu_buff1.data_dev_ptr(), blocks *
	                             sizeof(float),
	                             cudaMemcpyDeviceToHost, stream1));
	HANDLE_ERROR(cudaMemcpyAsync(data.host_buff2.data(),
	                             data.gpu_buff2.data_dev_ptr(), blocks *
	                             sizeof(float),
	                             cudaMemcpyDeviceToHost, stream1));

	Dt_gpu(data.g2, Vx, Vy, stream2);

	float sum1 = 0, sum2 = 0;

	HANDLE_ERROR(cudaStreamSynchronize(stream1));

	for (int i = 0; i < blocks; i++) {
		sum1 += data.host_buff1[i];
		sum2 += data.host_buff2[i];
	}

	data.lam2 = sum1;
	data.lam4 = sum2;

	data.f += (data.lam1 + data.beta / 2 * data.lam2 - data.lam4);
}

void update_mlp_gpu(tval3_data_gpu &data, const mat_device &b) {
	data.f += (data.lam4 + data.lam5);

	mat_device Vx(data.p, data.q, false);
	mat_device Vy(data.p, data.q, false);
	// Vx = Ux - Wx
	vec_add_gpu(data.n, 1,
	            data.Ux.data_dev_ptr(), (float)-1,
	            data.Wx.data_dev_ptr(), Vx.data_dev_ptr());
	// Vy = Uy - Wy
	vec_add_gpu(data.n, 1,
	            data.Uy.data_dev_ptr(), (float)-1,
	            data.Wy.data_dev_ptr(), Vy.data_dev_ptr());

	float val_alpha = -1 * data.beta;
	// sigmaX -= beta*Vx
	HANDLE_ERROR(cublasSaxpy(cb_handle, data.n, &val_alpha,
	                         Vx.data_dev_ptr(), 1,
	                         data.sigmaX.data_dev_ptr(), 1));
	// sigmaY -= beta*Vy
	HANDLE_ERROR(cublasSaxpy(cb_handle, data.n, &val_alpha,
	                         Vy.data_dev_ptr(), 1,
	                         data.sigmaY.data_dev_ptr(), 1));

	float result;
	HANDLE_ERROR(cublasSdot(cb_handle, data.n, data.sigmaX.data_dev_ptr(),
	                        1, Vx.data_dev_ptr(), 1, &data.lam4));
	HANDLE_ERROR(cublasSdot(cb_handle, data.n, data.sigmaY.data_dev_ptr(),
	                        1, Vy.data_dev_ptr(), 1, &result));
	data.lam4 += result;

	mat_device Aub(data.m, false);

	// Aub = Au - b
	vec_add_gpu(data.m, 1,
	            data.Au.data_dev_ptr(), (float)-1,
	            b.data_dev_ptr(), Aub.data_dev_ptr());
	val_alpha = -1 * data.mu;
	// delta -= mu*Aub
	HANDLE_ERROR(cublasSaxpy(cb_handle, data.m, &val_alpha,
	                         Aub.data_dev_ptr(), 1,
	                         data.delta.data_dev_ptr(), 1));
	HANDLE_ERROR(cublasSdot(cb_handle, data.m, data.delta.data_dev_ptr(), 1,
	                        Aub.data_dev_ptr(), 1, &data.lam5));

	data.f -= (data.lam4 + data.lam5);
}

template<class matrix_type>
const tval3_info tval3_gpu(mat_host &U, const matrix_type &A, const mat_host &b,
                           const tval3_options &opts, const mat_host &Ut,
                           bool pagelocked) {

	float elapsedTime;
	cudaEvent_t start_tval3, stop_tval3;
	HANDLE_ERROR(cudaEventCreate(&start_tval3));
	HANDLE_ERROR(cudaEventCreate(&stop_tval3));

	cudaEventRecord (start_tval3);

	if (majorRevision >= 2) {
		HANDLE_ERROR(cudaStreamCreate(&stream1));
		HANDLE_ERROR(cudaStreamCreate(&stream2));
		HANDLE_ERROR(cudaStreamCreate(&stream3));
	} else {
		stream1 = 0;
		stream2 = 0;
		stream3 = 0;
		channelDesc = cudaCreateChannelDesc<float>();
	}

	tval3_data_gpu data(U, b, opts.mu0, opts.beta0, pagelocked);
	mat_device b_gpu = b;

	float scl = scaleb_gpu(b, b_gpu);

	mat_device DtsAtd(data.n);
	// Atb = A'*b
	mat_vec_mul(CUBLAS_OP_T, A, b_gpu.data_dev_ptr(),
	            data.Atb.data_dev_ptr());

	mat_device U_gpu(data.p, data.q);
	// initial guess: U=A'*b
	U_gpu = data.Atb;

	float muf = opts.mu;
	float betaf = opts.beta;
	float beta0 = 0;
	float gam = opts.gam;

	if (data.mu > muf) data.mu = muf;

	if (data.beta > betaf) data.beta = betaf;
	data.muDbeta = data.mu / data.beta;

	mat_device rcdU = U_gpu, UrcdU(data.n, false);

	D_gpu(data.Ux, data.Uy, U_gpu);
	shrinkage_gpu(data);
	get_g_gpu(data, A, U_gpu, b_gpu);
	get_gradient_gpu(data, DtsAtd);

	float Q = 1, Qp;
	float C = data.f;
	int count_outer = 0, count_total = 0;
	bool fst_iter;
	float RelChg, RelChgOut = 0, nrmup = 0;

	while ((count_outer < opts.maxcnt) && (count_total < opts.maxit)) {

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
			            U_gpu.data_dev_ptr(), -1,
			            data.Up.data_dev_ptr(),
			            data.uup.data_dev_ptr());

			// compute gradient
			get_gradient_gpu(data, DtsAtd);

			count_total++;

			// calculate relative change
			HANDLE_ERROR(cublasSnrm2(cb_handle, data.n,
			                         data.Up.data_dev_ptr(), 1,
			                         &nrmup));


			HANDLE_ERROR(cublasSnrm2(cb_handle, data.n,
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
		vec_add_gpu(data.n, 1,
		            U_gpu.data_dev_ptr(), -1,
		            rcdU.data_dev_ptr(), UrcdU.data_dev_ptr());

		HANDLE_ERROR(cublasSnrm2(cb_handle, data.n,
		                         UrcdU.data_dev_ptr(), 1, &RelChgOut));
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
		float val_alpha = beta0 / data.beta;
		HANDLE_ERROR(cublasSscal(cb_handle, data.n, &val_alpha,
		                         DtsAtd.data_dev_ptr(), 1));

		// compute gradient
		get_gradient_gpu(data, DtsAtd);

		// reset reference values
		gam = opts.gam; Q = 1; C = data.f;

	}

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

	// generate return value
	tval3_info info;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_tval3,
	                                  stop_tval3));
	info.secs = elapsedTime / 1000;
	info.outer_iters = count_outer;
	info.total_iters = count_total;
	info.rel_chg = RelChgOut;

	if (Ut.len == data.n) {
		mat_device Ut_gpu = Ut;
		mat_device deltaU = Ut_gpu;
		float val_alpha = -1;
		HANDLE_ERROR(cublasSaxpy(cb_handle, data.n, &val_alpha,
		                         U_gpu.data_dev_ptr(), 1,
		                         deltaU.data_dev_ptr(), 1));
		float nrm_deltaU;
		HANDLE_ERROR(cublasSnrm2(cb_handle, data.n,
		                         deltaU.data_dev_ptr(), 1,
		                         &nrm_deltaU));
		float nrm_Ut;
		HANDLE_ERROR(cublasSnrm2(cb_handle, data.n,
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

	return info;
}

template<class mat_Type>
void check_params(mat_host &U, const mat_Type &A, const mat_host &b,
                  const mat_host &Ut, int layers) throw(tval3_gpu_exception) {

	if (U.format != mat_col_major || Ut.format != mat_col_major)
		throw tval3_gpu_exception(
		        "Argument error: Matrices must be in column major format!");


	else if (Ut.len > 0 && Ut.len != U.len)
		throw tval3_gpu_exception(
		        "Argument error: U and Ut must have the same size!");
	else if (U.len != A.cols * layers)
		throw tval3_gpu_exception(
		        "Argument error: the length of U must be equal to A.cols * layers!");


	else if (b.len != A.rows * layers)
		throw tval3_gpu_exception(
		        "Argument error: b.len must be equal to A.rows * layers!");

}

int getMajorRevision(int device) {
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
	return prop.major;
}

const tval3_info tval3_gpu(mat_host &U, const mat_host &A, const mat_host &b,
                           const tval3_options &opts, const mat_host &Ut,
                           bool pagelocked) {
	int cudaDevice;
	tval3_info info;

	try {
		if (A.format != mat_col_major)
			throw tval3_gpu_exception(
			        "Argument error: A must be in column major format!");






		check_params(U, A, b, Ut, b.len / A.rows);

		HANDLE_ERROR(cudaGetDevice(&cudaDevice));
		majorRevision = getMajorRevision(cudaDevice);

		HANDLE_ERROR(cublasCreate(&cb_handle));

		dense_mm mA(A, b.len / A.rows, cb_handle);

		info = tval3_gpu(U, mA, b, opts, Ut, pagelocked);

		HANDLE_ERROR(cublasDestroy(cb_handle));
	} catch (tval3_gpu_exception) {
		throw;
	} catch (const std::exception & ex) {
		std::string msg = "Internal error: ";
		msg += ex.what();
		throw tval3_gpu_exception(msg);
	}

	return info;
}

const tval3_info tval3_gpu(mat_host &U, const sparse_mat_host &A,
                           const mat_host &b,
                           const tval3_options &opts, const mat_host &Ut,
                           bool pagelocked) {
	int cudaDevice;
	tval3_info info;

	try {
		if (A.format != sparse_mat_csc)
			throw tval3_gpu_exception(
			        "Argument error: A must be in CSC format!");
		check_params(U, A, b, Ut, b.len / A.rows);

		HANDLE_ERROR(cudaGetDevice(&cudaDevice));
		majorRevision = getMajorRevision(cudaDevice);

		HANDLE_ERROR(cublasCreate(&cb_handle));
		HANDLE_ERROR(cusparseCreate(&cs_handle));
		// general matrix, zero based indexing
		HANDLE_ERROR(cusparseCreateMatDescr(&descrA));

		sparse_mm mA(A, b.len / A.rows, cs_handle, descrA);

		info = tval3_gpu(U, mA, b, opts, Ut, pagelocked);

		HANDLE_ERROR(cusparseDestroyMatDescr(descrA));
		HANDLE_ERROR(cusparseDestroy(cs_handle));
		HANDLE_ERROR(cublasDestroy(cb_handle));
	} catch (tval3_gpu_exception) {
		throw;
	} catch (const std::exception & ex) {
		std::string msg = "Internal error: ";
		msg += ex.what();
		throw tval3_gpu_exception(msg);
	}

	return info;
}


