#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "texture_fetch_functions.h"
#include "handle_error.h"
#include "tval3_gpu_3d_kernels.cuh"
#include <stdio.h>

#define PROFILING
//#define PROFILING0 // only for single device runs!!!
#include "profile_gpu.h"

// dynamically allocated shared memory array
extern __shared__ float fbuffer[];
extern __shared__ double dbuffer[];

// texture references...
texture<float, 1> texRef;
texture<float, 1> texRefX;
texture<float, 1> texRefY;
texture<float, 1> texRefZ;

texture<float, cudaTextureType3D, cudaReadModeElementType> texInput;
texture<int2, cudaTextureType3D, cudaReadModeElementType> texInput0;

//-----------------------------------------------------------------------------
// conversion functions: D2S and S2D
//-----------------------------------------------------------------------------
#if (1)

__global__
void convS2D_GPU_kernel (int N, double *output, float *input) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x);

	if (index < N) {
		output[index] = (double)input[index];
	}
}

void convS2D_GPU_kernel_wrapper(int blocks, int threads, size_t dynMem,
                                cudaStream_t stream,
                                int N, double *output, float *input) {

	convS2D_GPU_kernel << < blocks, threads, dynMem, stream >> >
	        (N, output, input);

}

__global__
void convD2S_GPU_kernel (int N, float *output, double *input) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x);

	if (index < N) {
		output[index] = (float)input[index];
	}
}

void convD2S_GPU_kernel_wrapper(int blocks, int threads, size_t dynMem,
                                cudaStream_t stream,
                                int N, float *output, double *input) {

	convD2S_GPU_kernel << < blocks, threads, dynMem, stream >> >
	        (N, output, input);

}

#endif
//-----------------------------------------------------------------------------
// vec_add_kernel: result = alpha*x + y + z
//-----------------------------------------------------------------------------
#if (1)

template<class Type>
__global__
void vec_add_kernel(int N, Type *result, const Type alpha, const Type *x,
                    const Type *y, const Type *z) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x);

	Type val;

	while (index < N) {
		val = alpha * x[index] + y[index];

		if (z != NULL)
			val += z[index];
		result[index] = val;
		index += gridDim.x * blockDim.x;
	}
}

template<class Type>
__global__
void vec_add_kernel(int N, Type *result, const Type alpha, const Type *x,
                    const Type *y) {

	int index = (threadIdx.x + blockIdx.x * blockDim.x);

	Type val;

	while (index < N) {
		val = alpha * x[index] + y[index];
		result[index] = val;
		index += gridDim.x * blockDim.x;
	}
}

void vec_add_kernel_wrapper(int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, float *result, const float alpha,
                            const float *x, const float *y, const float *z) {
#ifdef PROFILING
	profile_info[20].valid = true;
	sprintf(profile_info[20].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif

	if ( z != NULL)
		vec_add_kernel<float><< < blocks, threads, 0, stream >> >
		        (N, result, alpha, x, y, z);
	else
		vec_add_kernel<float><< < blocks, threads, 0, stream >> >
		        (N, result, alpha, x, y);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[20].time += elapsedTime;
	profile_info[20].runs++;
#endif
}

void vec_add_kernel_wrapper(int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, double *result, const double alpha,
                            const double *x, const double *y, const double *z) {
#ifdef PROFILING
	profile_info[20].valid = true;
	sprintf(profile_info[20].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif

	if ( z != NULL)
		vec_add_kernel<double><< < blocks, threads, 0, stream >> >
		        (N, result, alpha, x, y, z);
	else
		vec_add_kernel<double><< < blocks, threads, 0, stream >> >
		        (N, result, alpha, x, y);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[20].time += elapsedTime;
	profile_info[20].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------
// D_kernel_tex/_no_tex:
//-----------------------------------------------------------------------------
#if (1)

__global__
void D_kernel_no_tex(const float *U, float *Ux, float *Uy, float *Uz, int dim_x,
                     int dim_y, int dim_z) {

	int x, y, z, lin_index;

	float tmp;

	z = blockIdx.z * blockDim.z + threadIdx.z;
	while (z < dim_z) {
		y = blockIdx.y * blockDim.y + threadIdx.y;
		while (y < dim_y) {
			x = blockIdx.x * blockDim.x + threadIdx.x;

			while (x < dim_x) {
				lin_index = z * dim_x * dim_y + y * dim_x + x;

				tmp = U[lin_index];

				if (x < dim_x - 1)
					Ux[lin_index] = U[lin_index + 1] - tmp;
				else
					Ux[lin_index] = U[lin_index - x] - tmp;

				if (y < dim_y - 1)
					Uy[lin_index] =
					        U[lin_index + dim_x] - tmp;
				else
					Uy[lin_index] =
					        U[lin_index - y * dim_x] - tmp;

				if (z < dim_z - 1)
					Uz[lin_index] =
					        U[lin_index + dim_x *
					          dim_y] - tmp;
				else
					Uz[lin_index] =
					        U[lin_index - z * dim_x *
					          dim_y] - tmp;
				x += gridDim.x * blockDim.x;
			}
			y += gridDim.y * blockDim.y;
		}
		z += gridDim.z * blockDim.z;
	}
}

__global__
void D_kernel_tex(const float *U, float *Ux, float *Uy, float *Uz, int dim_x,
                  int dim_y, int dim_z) {

	int x, y, z, lin_index;

	float tmp;

	y = blockIdx.y * blockDim.y + threadIdx.y;
	while (y < dim_y) {
		x = blockIdx.x * blockDim.x + threadIdx.x;
		while (x < dim_x) {
			z = blockIdx.z * blockDim.z + threadIdx.z;
			while (z < dim_z) {
				lin_index = z * dim_x * dim_y + y * dim_x + x;

				tmp = tex1Dfetch(texRef, lin_index);

				if (x < dim_x - 1) {
					Ux[lin_index] = tex1Dfetch(texRef,
					                           lin_index +
					                           1) - tmp;
				} else {
					Ux[lin_index] = tex1Dfetch(texRef,
					                           lin_index -
					                           x) - tmp;
				}

				if (y < dim_y - 1) {
					Uy[lin_index] = tex1Dfetch(texRef,
					                           lin_index +
					                           dim_x) - tmp;
				} else {
					Uy[lin_index] = tex1Dfetch(texRef,
					                           lin_index - y *
					                           dim_x) - tmp;
				}

				if (z < dim_z - 1) {
					Uz[lin_index] = tex1Dfetch(texRef,
					                           lin_index + dim_x *
					                           dim_y) - tmp;
				} else {
					Uz[lin_index] = tex1Dfetch(texRef,
					                           lin_index - z * dim_x *
					                           dim_y) - tmp;
				}

				z += gridDim.z * blockDim.z;
			}
			x += gridDim.x * blockDim.x;
		}
		y += gridDim.y * blockDim.y;
	}
}

void D_kernel_no_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                             cudaStream_t stream,
                             const float *U, float *Ux, float *Uy, float *Uz,
                             int dim_x, int dim_y, int dim_z) {
#ifdef PROFILING
	profile_info[21].valid = true;
	sprintf(profile_info[21].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	D_kernel_no_tex << < blocks, threads, dynMem, stream >> >
	        (U, Ux, Uy, Uz, dim_x, dim_y, dim_z);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[21].time += elapsedTime;
	profile_info[21].runs++;
#endif
}

void D_kernel_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                          cudaStream_t stream, size_t texSize,
                          const float *U, float *Ux, float *Uy, float *Uz,
                          int dim_x, int dim_y, int dim_z) {
#ifdef PROFILING
	profile_info[22].valid = true;
	sprintf(profile_info[22].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	HANDLE_ERROR(cudaBindTexture(NULL, &texRef, U, &channelDesc, texSize));

	D_kernel_tex << < blocks, threads, dynMem, stream >> >
	        (U, Ux, Uy, Uz, dim_x, dim_y, dim_z);

	HANDLE_ERROR(cudaUnbindTexture(texRef));
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[22].time += elapsedTime;
	profile_info[22].runs++;
#endif
}

__global__
void D_kernel_no_tex(const double *U, double *Ux, double *Uy, double *Uz,
                     int dim_x, int dim_y, int dim_z) {

	int x, y, z, lin_index;

	double tmp;

	z = blockIdx.z * blockDim.z + threadIdx.z;
	while (z < dim_z) {
		y = blockIdx.y * blockDim.y + threadIdx.y;
		while (y < dim_y) {
			x = blockIdx.x * blockDim.x + threadIdx.x;

			while (x < dim_x) {
				lin_index = z * dim_x * dim_y + y * dim_x + x;

				tmp = U[lin_index];

				if (x < dim_x - 1)
					Ux[lin_index] = U[lin_index + 1] - tmp;
				else
					Ux[lin_index] = U[lin_index - x] - tmp;

				if (y < dim_y - 1)
					Uy[lin_index] =
					        U[lin_index + dim_x] - tmp;
				else
					Uy[lin_index] =
					        U[lin_index - y * dim_x] - tmp;

				if (z < dim_z - 1)
					Uz[lin_index] =
					        U[lin_index + dim_x *
					          dim_y] - tmp;
				else
					Uz[lin_index] =
					        U[lin_index - z * dim_x *
					          dim_y] - tmp;
				x += gridDim.x * blockDim.x;
			}
			y += gridDim.y * blockDim.y;
		}
		z += gridDim.z * blockDim.z;
	}
}

__global__
void D_kernel_tex(const double *U, double *Ux, double *Uy, double *Uz,
                  int dim_x, int dim_y, int dim_z) {

	int x, y, z, lin_index;

	double tmp;

	y = blockIdx.y * blockDim.y + threadIdx.y;
	while (y < dim_y) {
		x = blockIdx.x * blockDim.x + threadIdx.x;
		while (x < dim_x) {
			z = blockIdx.z * blockDim.z + threadIdx.z;
			while (z < dim_z) {
				lin_index = z * dim_x * dim_y + y * dim_x + x;

				tmp = tex1Dfetch(texRef, lin_index);

				if (x < dim_x - 1) {
					Ux[lin_index] = tex1Dfetch(texRef,
					                           lin_index +
					                           1) - tmp;
				} else {
					Ux[lin_index] = tex1Dfetch(texRef,
					                           lin_index -
					                           x) - tmp;
				}

				if (y < dim_y - 1) {
					Uy[lin_index] = tex1Dfetch(texRef,
					                           lin_index +
					                           dim_x) - tmp;
				} else {
					Uy[lin_index] = tex1Dfetch(texRef,
					                           lin_index - y *
					                           dim_x) - tmp;
				}

				if (z < dim_z - 1) {
					Uz[lin_index] = tex1Dfetch(texRef,
					                           lin_index + dim_x *
					                           dim_y) - tmp;
				} else {
					Uz[lin_index] = tex1Dfetch(texRef,
					                           lin_index - z * dim_x *
					                           dim_y) - tmp;
				}

				z += gridDim.z * blockDim.z;
			}
			x += gridDim.x * blockDim.x;
		}
		y += gridDim.y * blockDim.y;
	}
}

void D_kernel_no_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                             cudaStream_t stream,
                             const double *U, double *Ux, double *Uy,
                             double *Uz, int dim_x, int dim_y, int dim_z) {
#ifdef PROFILING
	profile_info[21].valid = true;
	sprintf(profile_info[21].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	D_kernel_no_tex << < blocks, threads, dynMem, stream >> >
	        (U, Ux, Uy, Uz, dim_x, dim_y, dim_z);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[21].time += elapsedTime;
	profile_info[21].runs++;
#endif
}

void D_kernel_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                          cudaStream_t stream, size_t texSize,
                          const double *U, double *Ux, double *Uy, double *Uz,
                          int dim_x, int dim_y, int dim_z) {
#ifdef PROFILING
	profile_info[22].valid = true;
	sprintf(profile_info[22].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<double>();
	HANDLE_ERROR(cudaBindTexture(NULL, &texRef, U, &channelDesc, texSize));

	D_kernel_tex << < blocks, threads, dynMem, stream >> >
	        (U, Ux, Uy, Uz, dim_x, dim_y, dim_z);

	HANDLE_ERROR(cudaUnbindTexture(texRef));
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[22].time += elapsedTime;
	profile_info[22].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------
// Dt_kernel_tex/_no_tex:
//-----------------------------------------------------------------------------
#if (1)
__global__
void Dt_kernel_no_tex(const float *X, const float *Y, const float *Z,
                      float *res, int dim_x, int dim_y, int dim_z) {

	int x, y, z, lin_index;
	float xp, yp, zp;

	z = blockIdx.z * blockDim.z + threadIdx.z;
	while (z < dim_z) {
		y = blockIdx.y * blockDim.y + threadIdx.y;
		while (y < dim_y) {
			x = blockIdx.x * blockDim.x + threadIdx.x;
			while (x < dim_x) {

				lin_index = z * dim_x * dim_y + y * dim_x + x;

				xp = (x == 0) ? X[lin_index + dim_x - 1] : X[lin_index - 1];
				yp = (y == 0) ? Y[lin_index + (dim_y - 1) * dim_x] : 
					Y[lin_index - dim_x];
				zp = (z == 0) ? Z[lin_index + (dim_z - 1) * dim_x *
					dim_y] : Z[lin_index - dim_x * dim_y];

				res[lin_index] = xp - X[lin_index] + yp - Y[lin_index] + zp -
					Z[lin_index];

				x += blockDim.x * gridDim.x;
			}
			y += blockDim.y * gridDim.y;
		}
		z += blockDim.z * gridDim.z;
	}
}

__global__
void Dt_kernel_tex(const float *X, const float *Y, const float *Z, float *res,
		int dim_x, int dim_y, int dim_z) {

	int x, y, z, lin_index;
	float xp, yp, zp;

	y = blockIdx.y * blockDim.y + threadIdx.y;
	while (y < dim_y) {
		x = blockIdx.x * blockDim.x + threadIdx.x;
		while (x < dim_x) {
			z = blockIdx.z * blockDim.z + threadIdx.z;
			while (z < dim_y) {

				lin_index = z * dim_x * dim_y + y * dim_x + x;

				xp = (x == 0) ? tex1Dfetch(texRefX, lin_index + dim_x - 1) : 
					tex1Dfetch(texRefX, lin_index - 1);
				yp = (y == 0) ? tex1Dfetch(texRefY, lin_index + (dim_y - 1) * dim_x) : 
					tex1Dfetch(texRefY, lin_index - dim_x);
				yp = (z == 0) ? tex1Dfetch(texRefZ, lin_index + (dim_z - 1) * dim_x *
						dim_y) : tex1Dfetch(texRefZ, lin_index - dim_x * dim_y);

				res[lin_index] = xp - tex1Dfetch(texRefX, lin_index) +
					yp - tex1Dfetch(texRefY, lin_index) +
					zp - tex1Dfetch(texRefZ, lin_index);

				z += blockDim.z * gridDim.z;
			}
			x += blockDim.x * gridDim.x;
		}
		y += blockDim.y * gridDim.y;
	}
}

void Dt_kernel_no_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
		cudaStream_t stream, const float *X, const float *Y, const float *Z,
		float *res, int dim_x, int dim_y, int dim_z) {
#ifdef PROFILING
	profile_info[23].valid = true;
	sprintf(profile_info[23].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	Dt_kernel_no_tex << < blocks, threads, dynMem, stream >> >
	        (X, Y, Z, res, dim_x, dim_y, dim_z);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[23].time += elapsedTime;
	profile_info[23].runs++;
#endif
}

void Dt_kernel_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
		cudaStream_t stream, size_t texSizeX, size_t texSizeY, size_t texSizeZ,
		const float *X, const float *Y, const float *Z,
		float *res, int dim_x, int dim_y, int dim_z) {
#ifdef PROFILING
	profile_info[24].valid = true;
	sprintf(profile_info[24].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	HANDLE_ERROR(cudaBindTexture(NULL, &texRefX, X, &channelDesc, texSizeX));
	HANDLE_ERROR(cudaBindTexture(NULL, &texRefY, Y, &channelDesc, texSizeY));
	HANDLE_ERROR(cudaBindTexture(NULL, &texRefZ, Z, &channelDesc, texSizeZ));

	Dt_kernel_tex << < blocks, threads, dynMem, stream >> >
	        (X, Y, Z, res, dim_x, dim_y, dim_z);

	HANDLE_ERROR(cudaUnbindTexture(texRefX));
	HANDLE_ERROR(cudaUnbindTexture(texRefY));
	HANDLE_ERROR(cudaUnbindTexture(texRefZ));
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[24].time += elapsedTime;
	profile_info[24].runs++;
#endif
}

__global__
void Dt_kernel_no_tex(const double *X, const double *Y, const double *Z,
		double *res, int dim_x, int dim_y, int dim_z) {

	int x, y, z, lin_index;
	double xp, yp, zp;

	z = blockIdx.z * blockDim.z + threadIdx.z;
	while (z < dim_z) {
		y = blockIdx.y * blockDim.y + threadIdx.y;
		while (y < dim_y) {
			x = blockIdx.x * blockDim.x + threadIdx.x;
			while (x < dim_x) {

				lin_index = z * dim_x * dim_y + y * dim_x + x;

				xp = (x == 0) ? X[lin_index + dim_x - 1] : X[lin_index - 1];
				yp = (y == 0) ? Y[lin_index + (dim_y - 1) * dim_x] : 
					Y[lin_index - dim_x];
				zp = (z == 0) ? Z[lin_index + (dim_z - 1) * dim_x * dim_y] : 
					Z[lin_index - dim_x * dim_y];

				res[lin_index] = xp - X[lin_index] + yp - Y[lin_index] + zp -
					Z[lin_index];

				x += blockDim.x * gridDim.x;
			}
			y += blockDim.y * gridDim.y;
		}
		z += blockDim.z * gridDim.z;
	}
}

__global__
void Dt_kernel_tex(const double *X, const double *Y, const double *Z,
                   double *res, int dim_x, int dim_y, int dim_z) {

	int x, y, z, lin_index;
	double xp, yp, zp;

	y = blockIdx.y * blockDim.y + threadIdx.y;
	while (y < dim_y) {
		x = blockIdx.x * blockDim.x + threadIdx.x;
		while (x < dim_x) {
			z = blockIdx.z * blockDim.z + threadIdx.z;
			while (z < dim_y) {

				lin_index = z * dim_x * dim_y + y * dim_x + x;

				xp = (x == 0) ? tex1Dfetch(texRefX, lin_index + dim_x - 1) : 
					tex1Dfetch(texRefX, lin_index - 1);
				yp = (y == 0) ? tex1Dfetch(texRefY, lin_index + (dim_y - 1) *
						dim_x) : tex1Dfetch(texRefY, lin_index - dim_x);
				yp = (z == 0) ? tex1Dfetch(texRefZ, lin_index + (dim_z - 1) * 
						dim_x * dim_y) : tex1Dfetch(texRefZ, lin_index - dim_x * dim_y);

				res[lin_index] = xp - tex1Dfetch(texRefX, lin_index) +
					yp - tex1Dfetch(texRefY, lin_index) + zp - 
					tex1Dfetch(texRefZ, lin_index);

				z += blockDim.z * gridDim.z;
			}
			x += blockDim.x * gridDim.x;
		}
		y += blockDim.y * gridDim.y;
	}
}

void Dt_kernel_no_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
		cudaStream_t stream, const double *X, const double *Y, const double *Z,
		double *res, int dim_x, int dim_y, int dim_z) {
#ifdef PROFILING
	profile_info[23].valid = true;
	sprintf(profile_info[23].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	Dt_kernel_no_tex << < blocks, threads, dynMem, stream >> >
	        (X, Y, Z, res, dim_x, dim_y, dim_z);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[23].time += elapsedTime;
	profile_info[23].runs++;
#endif
}

void Dt_kernel_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
		cudaStream_t stream, size_t texSizeX, size_t texSizeY, size_t texSizeZ,
		const double *X, const double *Y, const double *Z, double *res, 
		int dim_x, int dim_y, int dim_z) {
#ifdef PROFILING
	profile_info[24].valid = true;
	sprintf(profile_info[24].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<double>();
	HANDLE_ERROR(cudaBindTexture(NULL, &texRefX, X, &channelDesc, texSizeX));
	HANDLE_ERROR(cudaBindTexture(NULL, &texRefY, Y, &channelDesc, texSizeY));
	HANDLE_ERROR(cudaBindTexture(NULL, &texRefZ, Z, &channelDesc, texSizeZ));

	Dt_kernel_tex << < blocks, threads, dynMem, stream >> >
	        (X, Y, Z, res, dim_x, dim_y, dim_z);

	HANDLE_ERROR(cudaUnbindTexture(texRefX));
	HANDLE_ERROR(cudaUnbindTexture(texRefY));
	HANDLE_ERROR(cudaUnbindTexture(texRefZ));
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[24].time += elapsedTime;
	profile_info[24].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------
// lam2_4_kernel
//-----------------------------------------------------------------------------
#if (1)
__global__
void lam2_4_kernel(int N, const float *Vx, const float *Vy, const float *Vz,
		const float *sigmaX, const float *sigmaY, const float *sigmaZ, 
		float *tmp_lam2, float *tmp_lam4) {

	// pointers to dynamically allocated shared memory...
	float *lam2_buffer = fbuffer;
	float *lam4_buffer = fbuffer + blockDim.x;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float Vxi, Vyi, Vzi, sum_lam2 = 0, sum_lam4 = 0;

	while (index < N) {
		Vxi = Vx[index];
		Vyi = Vy[index];
		Vyi = Vz[index];
		sum_lam2 += Vxi * Vxi + Vyi * Vyi + Vzi * Vzi;
		sum_lam4 += Vxi * sigmaX[index] + Vyi * sigmaY[index] + Vzi *
		            sigmaZ[index];

		index += gridDim.x * blockDim.x;
	}

	lam2_buffer[threadIdx.x] = sum_lam2;
	lam4_buffer[threadIdx.x] = sum_lam4;

	__syncthreads();

	// reduction...

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

void lam2_4_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, const float *Vx, const float *Vy,
                            const float *Vz, const float *sigmaX,
                            const float *sigmaY, const float *sigmaZ,
                            float *tmp_lam2, float *tmp_lam4) {
#ifdef PROFILING
	profile_info[25].valid = true;
	sprintf(profile_info[25].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	lam2_4_kernel << < blocks, threads, dynMem, stream >> >
	        (N, Vx, Vy, Vz, sigmaX, sigmaY, sigmaZ, tmp_lam2, tmp_lam4);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[25].time += elapsedTime;
	profile_info[25].runs++;
#endif
}

__global__
void lam2_4_kernel(int N, const double *Vx, const double *Vy, const double *Vz,
                   const double *sigmaX,
                   const double *sigmaY, const double *sigmaZ, double *tmp_lam2,
                   double *tmp_lam4) {

	// pointers to dynamically allocated shared memory...
	double *lam2_buffer = dbuffer;
	double *lam4_buffer = dbuffer + blockDim.x;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	double Vxi, Vyi, Vzi, sum_lam2 = 0, sum_lam4 = 0;

	while (index < N) {
		Vxi = Vx[index];
		Vyi = Vy[index];
		Vyi = Vz[index];
		sum_lam2 += Vxi * Vxi + Vyi * Vyi + Vzi * Vzi;
		sum_lam4 += Vxi * sigmaX[index] + Vyi * sigmaY[index] + Vzi *
		            sigmaZ[index];

		index += gridDim.x * blockDim.x;
	}

	lam2_buffer[threadIdx.x] = sum_lam2;
	lam4_buffer[threadIdx.x] = sum_lam4;

	__syncthreads();

	// reduction...

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

void lam2_4_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, const double *Vx, const double *Vy,
                            const double *Vz, const double *sigmaX,
                            const double *sigmaY, const double *sigmaZ,
                            double *tmp_lam2, double *tmp_lam4) {
#ifdef PROFILING
	profile_info[25].valid = true;
	sprintf(profile_info[25].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	lam2_4_kernel << < blocks, threads, dynMem, stream >> >
	        (N, Vx, Vy, Vz, sigmaX, sigmaY, sigmaZ, tmp_lam2, tmp_lam4);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[25].time += elapsedTime;
	profile_info[25].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------
// lam3_5_kernel
//-----------------------------------------------------------------------------
#if (1)
__global__
void lam3_5_kernel(int N, const float *Au, const float *b, const float *delta,
                   float *tmp_lam3, float *tmp_lam5) {

	// pointers to dynamically allocated shared memory...
	float *lam3_buffer = fbuffer;
	float *lam5_buffer = fbuffer + blockDim.x;

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

	// reduction...

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

void lam3_5_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, const float *Au, const float *b,
                            const float *delta, float *tmp_lam3,
                            float *tmp_lam5) {
#ifdef PROFILING
	profile_info[26].valid = true;
	sprintf(profile_info[26].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	lam3_5_kernel << < blocks, threads, dynMem, stream >> >
	        (N, Au, b, delta, tmp_lam3, tmp_lam5);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[26].time += elapsedTime;
	profile_info[26].runs++;
#endif
}

__global__
void lam3_5_kernel(int N, const double *Au, const double *b,
                   const double *delta, double *tmp_lam3, double *tmp_lam5) {

	// pointers to dynamically allocated shared memory...
	double *lam3_buffer = dbuffer;
	double *lam5_buffer = dbuffer + blockDim.x;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	double Aub, sum_lam3 = 0, sum_lam5 = 0;

	while (index < N) {
		Aub = Au[index] - b[index];
		sum_lam3 += Aub * Aub;
		sum_lam5 += delta[index] * Aub;

		index += gridDim.x * blockDim.x;
	}

	lam3_buffer[threadIdx.x] = sum_lam3;
	lam5_buffer[threadIdx.x] = sum_lam5;

	__syncthreads();

	// reduction...

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

void lam3_5_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, const double *Au, const double *b,
                            const double *delta, double *tmp_lam3,
                            double *tmp_lam5) {
#ifdef PROFILING
	profile_info[26].valid = true;
	sprintf(profile_info[26].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	lam3_5_kernel << < blocks, threads, dynMem, stream >> >
	        (N, Au, b, delta, tmp_lam3, tmp_lam5);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[26].time += elapsedTime;
	profile_info[26].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------
// get_tau_kernel
//-----------------------------------------------------------------------------
#if (1)
__global__
void get_tau_kernel(int N, const float *g, const float *gp, const float *g2,
                    const float *g2p, const float *uup,
                    float muDbeta, float *dg, float *dg2, float *tmp_ss,
                    float *tmp_sy) {

	// pointers to dynamically allocated shared memory...
	float *ss = fbuffer;
	float *sy = fbuffer + blockDim.x;

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

	// reduction...

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

void get_tau_kernel_wrapper( int blocks, int threads, size_t dynMem,
                             cudaStream_t stream,
                             int N, const float *g, const float *gp,
                             const float *g2, const float *g2p,
                             const float *uup,
                             float muDbeta, float *dg, float *dg2,
                             float *tmp_ss, float *tmp_sy) {
#ifdef PROFILING
	profile_info[27].valid = true;
	sprintf(profile_info[27].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	get_tau_kernel << < blocks, threads, dynMem, stream >> >
	        (N, g, gp, g2, g2p, uup, muDbeta, dg, dg2, tmp_ss, tmp_sy);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[27].time += elapsedTime;
	profile_info[27].runs++;
#endif
}

__global__
void get_tau_kernel(int N, const double *g, const double *gp, const double *g2,
                    const double *g2p, const double *uup,
                    double muDbeta, double *dg, double *dg2, double *tmp_ss,
                    double *tmp_sy) {

	// pointers to dynamically allocated shared memory...
	double *ss = dbuffer;
	double *sy = dbuffer + blockDim.x;

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double uup_i, dg_i, dg2_i, sum_ss = 0, sum_sy = 0;

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

	// reduction...

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

void get_tau_kernel_wrapper( int blocks, int threads, size_t dynMem,
                             cudaStream_t stream,
                             int N, const double *g, const double *gp,
                             const double *g2, const double *g2p,
                             const double *uup,
                             double muDbeta, double *dg, double *dg2,
                             double *tmp_ss, double *tmp_sy) {
#ifdef PROFILING
	profile_info[27].valid = true;
	sprintf(profile_info[27].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	get_tau_kernel << < blocks, threads, dynMem, stream >> >
	        (N, g, gp, g2, g2p, uup, muDbeta, dg, dg2, tmp_ss, tmp_sy);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[27].time += elapsedTime;
	profile_info[27].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------
// nonneg_kernel
//-----------------------------------------------------------------------------
#if (1)
__global__
void nonneg_kernel(int N, float *vec) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	float val;
	while (index < N) {
		val = vec[index];
		vec[index] = (val < 0) ? 0 : val;

		index += gridDim.x * blockDim.x;
	}
}

void nonneg_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, float *vec) {
#ifdef PROFILING
	profile_info[28].valid = true;
	sprintf(profile_info[28].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	nonneg_kernel << < blocks, threads, dynMem, stream >> > (N, vec);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[28].time += elapsedTime;
	profile_info[28].runs++;
#endif
}

__global__
void nonneg_kernel(int N, double *vec) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double val;
	while (index < N) {
		val = vec[index];
		vec[index] = (val < 0) ? 0 : val;

		index += gridDim.x * blockDim.x;
	}
}

void nonneg_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, double *vec) {
#ifdef PROFILING
	profile_info[28].valid = true;
	sprintf(profile_info[28].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	nonneg_kernel << < blocks, threads, dynMem, stream >> > (N, vec);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[28].time += elapsedTime;
	profile_info[28].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------
// shrinkage_kernel
//-----------------------------------------------------------------------------
#if (1)
__global__
void shrinkage_kernel(int N, const float *Ux, const float *Uy, const float *Uz,
                      const float *sigmaX,
                      const float *sigmaY, const float *sigmaZ, float beta,
                      float *Wx, float *Wy, float *Wz, float *tmp) {

	float Uxbar, Uybar, Uzbar, temp_wx, temp_wy, temp_wz;

	// pointer to dynamically allocated shared memory...
	float *buff = fbuffer;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0;

	while (index < N) {
		Uxbar = Ux[index] - sigmaX[index] / beta;
		Uybar = Uy[index] - sigmaY[index] / beta;
		Uzbar = Uz[index] - sigmaZ[index] / beta;
		temp_wx = abs(Uxbar) - 1 / beta;
		temp_wy = abs(Uybar) - 1 / beta;
		temp_wz = abs(Uzbar) - 1 / beta;
		temp_wx = (temp_wx >= 0) ? temp_wx : 0;
		temp_wy = (temp_wy >= 0) ? temp_wy : 0;
		temp_wz = (temp_wz >= 0) ? temp_wz : 0;
		Wx[index] = (Uxbar >= 0) ? temp_wx : -temp_wx;
		Wy[index] = (Uybar >= 0) ? temp_wy : -temp_wy;
		Wz[index] = (Uzbar >= 0) ? temp_wz : -temp_wz;

		sum += temp_wx + temp_wy + temp_wz;

		index += gridDim.x * blockDim.x;
	}

	buff[threadIdx.x] = sum;

	__syncthreads();

	// reduction...

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

void shrinkage_kernel_wrapper( int blocks, int threads, size_t dynMem,
                               cudaStream_t stream,
                               int N, const float *Ux, const float *Uy,
                               const float *Uz, const float *sigmaX,
                               const float *sigmaY, const float *sigmaZ,
                               float beta, float *Wx, float *Wy, float *Wz,
                               float *tmp) {
#ifdef PROFILING
	profile_info[29].valid = true;
	sprintf(profile_info[29].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	shrinkage_kernel << < blocks, threads, dynMem, stream >> >
	        (N, Ux, Uy, Uz, sigmaX, sigmaY, sigmaZ, beta, Wx, Wy, Wz, tmp);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[29].time += elapsedTime;
	profile_info[29].runs++;
#endif
}

__global__
void shrinkage_kernel(int N, const double *Ux, const double *Uy,
                      const double *Uz, const double *sigmaX,
                      const double *sigmaY, const double *sigmaZ, double beta,
                      double *Wx, double *Wy, double *Wz, double *tmp) {

	double Uxbar, Uybar, Uzbar, temp_wx, temp_wy, temp_wz;

	// pointer to dynamically allocated shared memory...
	double *buff = dbuffer;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0;

	while (index < N) {
		Uxbar = Ux[index] - sigmaX[index] / beta;
		Uybar = Uy[index] - sigmaY[index] / beta;
		Uzbar = Uz[index] - sigmaZ[index] / beta;
		temp_wx = abs(Uxbar) - 1 / beta;
		temp_wy = abs(Uybar) - 1 / beta;
		temp_wz = abs(Uzbar) - 1 / beta;
		temp_wx = (temp_wx >= 0) ? temp_wx : 0;
		temp_wy = (temp_wy >= 0) ? temp_wy : 0;
		temp_wz = (temp_wz >= 0) ? temp_wz : 0;
		Wx[index] = (Uxbar >= 0) ? temp_wx : -temp_wx;
		Wy[index] = (Uybar >= 0) ? temp_wy : -temp_wy;
		Wz[index] = (Uzbar >= 0) ? temp_wz : -temp_wz;

		sum += temp_wx + temp_wy + temp_wz;

		index += gridDim.x * blockDim.x;
	}

	buff[threadIdx.x] = sum;

	__syncthreads();

	// reduction...

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

void shrinkage_kernel_wrapper( int blocks, int threads, size_t dynMem,
                               cudaStream_t stream,
                               int N, const double *Ux, const double *Uy,
                               const double *Uz, const double *sigmaX,
                               const double *sigmaY, const double *sigmaZ,
                               double beta, double *Wx, double *Wy, double *Wz,
                               double *tmp) {
#ifdef PROFILING
	profile_info[29].valid = true;
	sprintf(profile_info[29].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	shrinkage_kernel << < blocks, threads, dynMem, stream >> >
	        (N, Ux, Uy, Uz, sigmaX, sigmaY, sigmaZ, beta, Wx, Wy, Wz, tmp);
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[29].time += elapsedTime;
	profile_info[29].runs++;
#endif
}
#endif

//-----------------------------------------------------------------------------
// routines for dynamic calculation of measurement matrix: atomicAdd_cc1 + mult_element + dyn_calc_kernel
//-----------------------------------------------------------------------------
__device__ inline
float atomicAdd_cc1(float* address, float val) {

	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;

	do {
		assumed = old;
		old =
		        atomicCAS(address_as_int, assumed,
		                  __float_as_int(val +
		                                 __int_as_float(assumed)));
	} while (assumed != old);

	return __int_as_float(old);
}

__device__ inline
double atomicAdd_cc1(double* address, double val) {

	unsigned long long oldval, newval, readback;

	oldval = __double_as_longlong(*address);
	newval = __double_as_longlong(__longlong_as_double(oldval) + val);

	while ((readback =
	                atomicCAS((unsigned long long *)address, oldval,
	                          newval)) != oldval)
	{
		oldval = readback;
		newval =
		        __double_as_longlong(__longlong_as_double(oldval) +
		                             val);
	}

	return __longlong_as_double(oldval);
}

__device__ inline
float myAtomicAdd(float* address, float val) {

	return atomicAdd(address, val);

}

__device__ inline
double myAtomicAdd(double* address, double val) {

	return atomicAdd_cc1(address, val);

}

__device__ inline
void mult_elementN(cublasOperation_t transA, int ray, int x_pixel, int y_pixel,
                   int z_pixel, int dim_x, int dim_y, float scale_factor,
                   const float *x, float *y) {

	float xC = y_pixel;
	float yC = x_pixel;
	float zC = z_pixel;

	*y += tex3D(texInput, xC, yC, zC) * scale_factor;

}

__device__ inline
void mult_elementN(cublasOperation_t transA, int ray, int x_pixel, int y_pixel,
                   int z_pixel, int dim_x, int dim_y, double scale_factor,
                   const double *x, double *y) {

	//int lin_index = z_pixel * (dim_x*dim_y) + x_pixel * dim_y + y_pixel; // col-major: y-fast

	//*y += x[lin_index] * scale_factor;

	float xC = y_pixel;
	float yC = x_pixel;
	float zC = z_pixel;

	int2 tmp = tex3D(texInput0, xC, yC, zC);

	double *tmp2 = (double*)&tmp;

	*y += (double)(tmp2[0]) * scale_factor;

}

template<class Type>
__device__ inline
void mult_elementT(cublasOperation_t transA, int ray, int x_pixel, int y_pixel,
                   int z_pixel, int dim_x, int dim_y, Type scale_factor,
                   const Type *x, Type *y) {

	int lin_index = z_pixel * (dim_x * dim_y) + x_pixel * dim_y + y_pixel; // col-major: y-fast

	//y[lin_index] += x[ray] * scale_factor;

	myAtomicAdd(y + lin_index, x[ray] * scale_factor);

}

template <unsigned int sortSize, class Type, class Type2>
__global__
void sum_vol_up_kernel(int dim_x, int dim_y, int dim_z, Type *y,
                       const Type * y_local) {

	int vol_z = blockIdx.z * blockDim.z + threadIdx.z;
	int vol_x = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int sub_vol = sortSize / 2;

	int wr_vol_y = blockIdx.x *
	               (blockDim.x / sub_vol) + threadIdx.x / sub_vol;              // for write alternative: remove division of threadIDx.x by sub_vol
	int wrIdx = vol_z * (dim_x * dim_y) + vol_x * dim_y + wr_vol_y;

	int rd_vol_y = blockIdx.x * blockDim.x + threadIdx.x;
	int rdIdx = vol_z *
	            (dim_x * dim_y *
	             sub_vol) + vol_x * dim_y * sub_vol + rd_vol_y;

	int idx = threadIdx.x % sub_vol;

	Type *buffer = (Type*)fbuffer;

	// 1st step: load all sub_vol values into shared mem
	Type2 tmp = ((Type2*)y_local)[rdIdx];
	buffer[threadIdx.x] = tmp.x + tmp.y;

	if (sortSize > 32)
		__syncthreads();

	// 2nd step: accumulate values in parallel reduction tree: active thread number halves in each iteration
	#pragma unroll 8
	for (unsigned int i = sortSize / 4; i > 0; i >>= 1 ) {

		if ( idx < i)
			buffer[threadIdx.x] += buffer[threadIdx.x + i];

		if (sortSize > 32)
			__syncthreads();

	}

	__syncthreads();

	// 3rd step: only final thread updates result vector
	//if (threadIdx.x < N)
	//	y[wrIdx] = buffer[threadIdx.x*sub_vol];
	if (idx == 0)
		y[wrIdx] = buffer[threadIdx.x];

}

template <unsigned int sortSize, bool accResult, class Type>
__global__
void sum_vol_up_kernel_inter(int N, int dim_x, int dim_y, int dim_z, Type *y,
                             const Type * y_local) {

	int vol_size = dim_x * dim_y * dim_z;

	int vol_z = blockIdx.z * blockDim.z + threadIdx.z; // blockDim.z = 1 --> = blockIdx.z
	int vol_y = blockIdx.y * blockDim.y + threadIdx.y; // blockDim.y = 1 --> = threadIdx.y

	int pixelPerBlock = blockDim.x / sortSize;
	int vol_x = blockIdx.x * pixelPerBlock + (threadIdx.x % pixelPerBlock);
	int vol_idx = vol_z * (dim_x * dim_y) + vol_y * dim_x + vol_x;

	int idx = threadIdx.x / pixelPerBlock;
	int rdIdx = vol_idx + idx * vol_size;

	Type *buffer = (Type*)fbuffer;

	// 1st step: load all sub_vol values into shared mem
	Type tmp = 0;
	while (rdIdx < N * vol_size) {
		tmp += y_local[rdIdx];
		rdIdx += sortSize * vol_size;
	}
	buffer[threadIdx.x] = tmp;

	__syncthreads();

	/// 2nd step: accumulate values in parallel reduction tree: active thread number halves in each iteration
	#pragma unroll 8
	for (unsigned int i = sortSize / 2; i > 0; i >>= 1 ) {

		if ( idx < i)
			buffer[threadIdx.x] +=
			        buffer[threadIdx.x + i * pixelPerBlock];

		__syncthreads();

	}

	// 3rd step: only final thread updates result vector
	if (idx == 0) {
		if ( accResult == true)
			y[vol_idx] += buffer[threadIdx.x];
		else
			y[vol_idx] = buffer[threadIdx.x];
	}

}

template<unsigned int subSize, class Type>
__global__
void dyn_calc_kernel(cublasOperation_t transA, int num_emitters,
                     int num_receivers, int rv_x, int rv_y, int rv_z,
                     Type scale_factor, const int *x_receivers,
                     const int *y_receivers, const int *z_receivers,
                     const int *x_emitters,
                     const int *y_emitters, const int *z_emitters,
                     const Type *x, Type *y, const unsigned int *use_path,
                     const unsigned int emitter0) {

	int d1, d2, d3, inc1, m1, inc2, m2, inc3, m3, x_pixel, y_pixel, z_pixel,
	    ray, err_1, err_2;
	int *pixel1, *pixel2, *pixel3;
	int emitter, receiver;
	int y_inc, y_mul;

	if (transA == CUBLAS_OP_N) {
		y_mul = 1;
		y_inc = 0;
	} else {
		y_mul = subSize;

		if ( subSize < 32)
			y_inc = (threadIdx.x % subSize);
		else
			y_inc =
			        (blockIdx.y %
			         (subSize / 32)) * 32 + (threadIdx.x % 32);

	}

	Type y_loc;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint2 tmp;

	if (idx < emitter0 ) {

		tmp = ((uint2*)use_path)[idx];
		emitter = tmp.x;
		receiver = tmp.y;
		ray = idx;

		// start at current emitter
		x_pixel = x_emitters[emitter];
		y_pixel = y_emitters[emitter];
		z_pixel = z_emitters[emitter];

		if (transA == CUBLAS_OP_N)
			mult_elementN(transA, ray, x_pixel,
			              y_pixel * y_mul + y_inc, z_pixel, rv_x,
			              rv_y * y_mul, scale_factor, x, &y_loc);
		else
			mult_elementT(transA, ray, x_pixel,
			              y_pixel * y_mul + y_inc, z_pixel, rv_x,
			              rv_y * y_mul, scale_factor, x, y);


		// calculate x,z,z distances to current receiver
		d1 = x_receivers[receiver] - x_pixel;
		d2 = y_receivers[receiver] - y_pixel;
		d3 = z_receivers[receiver] - z_pixel;

		// determine fastest (longest) dimension of path
		m1 = abs(d1); m2 = abs(d2); m3 = abs(d3);

		if (m1 >= m2 && m1 >= m3) { // (dx >= dy) & (dx >= dz) --> x is fastest: everything ok!

			pixel1 = &x_pixel; //x
			pixel2 = &y_pixel; //y
			pixel3 = &z_pixel; //z

			//d1: x, d2: y, d3: z

		} else if (m2 >= m1 && m2 >= m3) {  // (dy >= dx) & (dy >= dz) --> y is fastest: switch x and y distances!

			int tmp = d1;
			d1 = d2;
			d2 = tmp;

			pixel1 = &y_pixel;
			pixel2 = &x_pixel;
			pixel3 = &z_pixel;

			//d1: y, d2: x, d3: z

		} else { // (dz > dx) & (dz > dy) --> z is fastest: switch x and z distances!

			int tmp = d1;
			d1 = d3;
			d3 = d2;
			d2 = tmp;

			pixel1 = &z_pixel;
			pixel2 = &x_pixel;
			pixel3 = &y_pixel;

			//d1: z, d2: x, d3: y
		}

		// walk in positive or negative direction?
		inc1 = (d1 < 0) ? -1 : 1;
		inc2 = (d2 < 0) ? -1 : 1;
		inc3 = (d3 < 0) ? -1 : 1;

		// re-determine path length (may be changed by switching of axis)
		m1 = abs(d1);
		m2 = abs(d2);
		m3 = abs(d3);

		// set start error values
		err_1 = 2 * m2 - m1;
		err_2 = 2 * m3 - m1;

		// walk until longest dimension is reached
		for (int j = 1; j < m1 + 1; j++) {

			// step in 2nd direction necessary?
			if (err_1 > 0) {
				*pixel2 += inc2;
				err_1 -= 2 * m1;
			}

			// step in 3rd direction necessary?
			if (err_2 > 0) {
				*pixel3 += inc3;
				err_2 -= 2 * m1;
			}

			//always step in fastest direction: update errors and do step
			err_1 += 2 * m2;
			err_2 += 2 * m3;
			*pixel1 += inc1;

			//do matrix multiplication for current pixel
			if (transA == CUBLAS_OP_N)
				mult_elementN(transA, ray, x_pixel,
				              y_pixel * y_mul + y_inc, z_pixel,
				              rv_x, rv_y * y_mul, scale_factor,
				              x, &y_loc);
			else
				mult_elementT(transA, ray, x_pixel,
				              y_pixel * y_mul + y_inc, z_pixel,
				              rv_x, rv_y * y_mul, scale_factor,
				              x, y);

		}

		if (transA == CUBLAS_OP_N)
			y[ray] = y_loc;
	}
}

// old version: T-multiply without subvolumes
#if (0)
__global__
void dyn_calc_kernel(cublasOperation_t transA, int num_emitters,
                     int num_receivers, int rv_x, int rv_y, int rv_z,
                     double scale_factor, const int *x_receivers,
                     const int *y_receivers, const int *z_receivers,
                     const int *x_emitters,
                     const int *y_emitters, const int *z_emitters,
                     const double *x, double *y,
                     const unsigned short *use_path) {


	/* old version 
	   int index_r, index_e, inc_r, inc_e;
	   int d1, d2, d3, inc1, m1, inc2, m2, inc3, m3, x_pixel, y_pixel, z_pixel, ray, err_1, err_2;
	   int *pixel1, *pixel2, *pixel3;

	   // different assignment of paths to threads for multiplication with or without transposition
	   // (performance reasons)
	   // no transposition: x-> receivers, y-> emitters
	   // transposition: x-> emitters, y-> receivers
	   if(transA == CUBLAS_OP_N) {
	        //          0 .. 63   +    0..21   *    64
	        index_r = threadIdx.x + blockIdx.x * blockDim.x; // 0 .. 1407
	        //          0  .. 3   +    0..8    *     4
	        index_e = threadIdx.y + blockIdx.y * blockDim.y; // 0 .. 35
	        //         64      *    22
	        inc_r = blockDim.x * gridDim.x; // 1408
	        //          4      *     9
	        inc_e = blockDim.y * gridDim.y; // 36
	   } else {
	        index_e = threadIdx.x + blockIdx.x * blockDim.x;
	        index_r = threadIdx.y + blockIdx.y * blockDim.y;
	        inc_e = blockDim.x * gridDim.x;
	        inc_r = blockDim.y * gridDim.y;
	   }

	   // trace path using Bresenham algorithm...
	   for(int receiver=index_r; receiver < num_receivers; receiver+= inc_r) {
	        for(int emitter=index_e; emitter < num_emitters; emitter+= inc_e) {

	 */

	int d1, d2, d3, inc1, m1, inc2, m2, inc3, m3, x_pixel, y_pixel, z_pixel,
	    ray, err_1, err_2;
	int *pixel1, *pixel2, *pixel3;
	int emitter, receiver;
	int y_inc, y_mul;

	if (transA == CUBLAS_OP_N) {
		emitter = blockIdx.z * gridDim.y + blockIdx.y;    // 0 .. 627
		receiver = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. 1413
		y_inc = 0;
		y_mul = 1;
	} else {
		emitter = blockIdx.z * gridDim.y + blockIdx.y;    // 0 .. 627
		receiver = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. 1413
		y_inc = (threadIdx.x % 32);
		y_mul = 32;
	}

	// trace path from emitter to receiver using Bresenham's algorithm...
	if ((receiver < num_receivers) & (emitter < num_emitters)) {

		// number of ray (element in y-vector)
		ray = emitter * num_receivers + receiver;

		//if ( use_path[ray] == 0)
		//continue;

		if ( use_path[ray] != 0) {

			// start at current emitter
			x_pixel = x_emitters[emitter];
			y_pixel = y_emitters[emitter];
			z_pixel = z_emitters[emitter];

			mult_element(transA, ray, x_pixel,
			             y_pixel * y_mul + y_inc, z_pixel, rv_x,
			             rv_y * y_mul, scale_factor, x, y);

			// calculate x,z,z distances to current receiver
			d1 = x_receivers[receiver] - x_pixel;
			d2 = y_receivers[receiver] - y_pixel;
			d3 = z_receivers[receiver] - z_pixel;

			// determine fastest (longest) dimension of path
			m1 = abs(d1); m2 = abs(d2); m3 = abs(d3);

			if (m1 >= m2 && m1 >= m3) { // (dx >= dy) & (dx >= dz) --> x is fastest: everything ok!

				pixel1 = &x_pixel; //x
				pixel2 = &y_pixel; //y
				pixel3 = &z_pixel; //z

				//d1: x, d2: y, d3: z

			} else if (m2 >= m1 && m2 >= m3) {  // (dy >= dx) & (dy >= dz) --> y is fastest: switch x and y distances!

				int tmp = d1;
				d1 = d2;
				d2 = tmp;

				pixel1 = &y_pixel;
				pixel2 = &x_pixel;
				pixel3 = &z_pixel;

				//d1: y, d2: x, d3: z

			} else { // (dz > dx) & (dz > dy) --> z is fastest: switch x and z distances!

				int tmp = d1;
				d1 = d3;
				d3 = d2;
				d2 = tmp;

				pixel1 = &z_pixel;
				pixel2 = &x_pixel;
				pixel3 = &y_pixel;

				//d1: z, d2: x, d3: y
			}

			// walk in positive or negative direction?
			inc1 = (d1 < 0) ? -1 : 1;
			inc2 = (d2 < 0) ? -1 : 1;
			inc3 = (d3 < 0) ? -1 : 1;

			// re-determine path length (may be changed by switching of axis)
			m1 = abs(d1);
			m2 = abs(d2);
			m3 = abs(d3);

			// set start error values
			err_1 = 2 * m2 - m1;
			err_2 = 2 * m3 - m1;

			// walk until longest dimension is reached
			for (int j = 1; j < m1 + 1; j++) {

				// step in 2nd direction necessary?
				if (err_1 > 0) {
					*pixel2 += inc2;
					err_1 -= 2 * m1;
				}

				// step in 3rd direction necessary?
				if (err_2 > 0) {
					*pixel3 += inc3;
					err_2 -= 2 * m1;
				}

				//always step in fastest direction: update errors and do step
				err_1 += 2 * m2;
				err_2 += 2 * m3;
				*pixel1 += inc1;

				//do matrix multiplication for current pixel
				mult_element(transA, ray, x_pixel,
				             y_pixel * y_mul + y_inc, z_pixel,
				             rv_x, rv_y * y_mul, scale_factor,
				             x, y);

			}
		}
	}
}
#endif

//pixel-wise T-multiply: takes years!!!
#if (0)
template<class Type>
__global__
void dyn_calc_kernelT(cublasOperation_t transA, int num_emitters,
                      int num_receivers, int rv_x, int rv_y, int rv_z,
                      Type scale_factor, const int *x_receivers,
                      const int *y_receivers, const int *z_receivers,
                      const int *x_emitters,
                      const int *y_emitters, const int *z_emitters,
                      const Type *x, Type *y, const unsigned short *use_path) {

	int d1, d2, d3, inc1, m1, inc2, m2, inc3, m3, x_pixel, y_pixel, z_pixel,
	    ray, err_1, err_2;
	int *pixel1, *pixel2, *pixel3;
	int emitter, receiver, emitterStart, emitterInc;

	int max = 0;

	int wrIdx = blockIdx.z *
	            (gridDim.x *
	             gridDim.y) + blockIdx.x * gridDim.y + blockIdx.x;

	int x0 = blockIdx.x;
	int y0 = blockIdx.y;
	int z0 = blockIdx.z;

	emitterStart = threadIdx.x;
	emitterInc = blockDim.x;

	Type value = 0;

/*	Type deltaX0, deltaY0, deltaZ0;
        Type deltaX1, deltaY1, deltaZ1;

        Type lambdaX, lambdaY, lambdaZ;

        float epsilon = 0.999995;*/

	for (emitter = emitterStart;
	     emitter < num_emitters;
	     emitter += emitterInc) {
		for (receiver = 0; receiver < num_receivers; receiver++) {

			// number of ray (element in y-vector)
			ray = emitter * num_receivers + receiver;

			if ( use_path[ray] != 0) {

				/*
				   // current emitter
				   xe = x_emitters[emitter];
				   ye = y_emitters[emitter];
				   ze = z_emitters[emitter];

				   // current receiver
				   xr = x_receivers[receiver];
				   yr = y_receivers[receiver];
				   zr = z_receivers[receiver];


				   deltaX0 = xr-xe;
				   deltaY0 = yr-ye;
				   deltaZ0 = zr-ze;

				   deltaX1 = x_pixel-xr;
				   deltaY1 = y_pixel-yr;
				   deltaZ1 = z_pixel-zr;

				   lambdaX = deltaX0 / deltaX1;
				   lambdaY = deltaY0 / deltaY1;
				   lambdaZ = deltaZ0 / deltaZ1;

				   if ( ((lambdaX > 0 ) & (lambdaY > 0) & (lambdaZ > 0)) &
				                ((lambdaX * epsilon < lambdaY) & (lambdaY < lambdaX * 1/epsilon)) &
				                ((lambdaZ * epsilon < lambdaY) & (lambdaY < lambdaZ * 1/epsilon) ) )

				        value += x[ray] * scale_factor;*/

				// current emitter
				x_pixel = x_emitters[emitter];
				y_pixel = y_emitters[emitter];
				z_pixel = z_emitters[emitter];

				// calculate x,y,z distances to current receiver
				d1 = x_receivers[receiver] - x_pixel;
				d2 = y_receivers[receiver] - y_pixel;
				d3 = z_receivers[receiver] - z_pixel;

				// determine fastest (longest) dimension of path
				m1 = abs(d1); m2 = abs(d2); m3 = abs(d3);

				if (m1 >= m2 && m1 >= m3) { // (dx >= dy) & (dx >= dz) --> x is fastest: everything ok!

					max = x0;
					pixel1 = &x_pixel; //x
					pixel2 = &y_pixel; //y
					pixel3 = &z_pixel; //z

					//d1: x, d2: y, d3: z

				} else if (m2 >= m1 && m2 >= m3) {  // (dy >= dx) & (dy >= dz) --> y is fastest: switch x and y distances!

					int tmp = d1;
					d1 = d2;
					d2 = tmp;

					max = y0;
					pixel1 = &y_pixel;
					pixel2 = &x_pixel;
					pixel3 = &z_pixel;

					//d1: y, d2: x, d3: z

				} else { // (dz > dx) & (dz > dy) --> z is fastest: switch x and z distances!

					int tmp = d1;
					d1 = d3;
					d3 = d2;
					d2 = tmp;

					max = z0;
					pixel1 = &z_pixel;
					pixel2 = &x_pixel;
					pixel3 = &y_pixel;

					//d1: z, d2: x, d3: y
				}

				// walk in positive or negative direction?
				inc1 = (d1 < 0) ? -1 : 1;
				inc2 = (d2 < 0) ? -1 : 1;
				inc3 = (d3 < 0) ? -1 : 1;

				// re-determine path length (may be changed by switching of axis)
				m1 = abs(d1);
				m2 = abs(d2);
				m3 = abs(d3);

				// set start error values
				err_1 = 2 * m2 - m1;
				err_2 = 2 * m3 - m1;

				// walk until longest dimension is reached
				for (int j = 1; j < m1 + 1; j++) {

					// step in 2nd direction necessary?
					if (err_1 > 0) {
						*pixel2 += inc2;
						err_1 -= 2 * m1;
					}

					// step in 3rd direction necessary?
					if (err_2 > 0) {
						*pixel3 += inc3;
						err_2 -= 2 * m1;
					}

					//always step in fastest direction: update errors and do step
					err_1 += 2 * m2;
					err_2 += 2 * m3;
					*pixel1 += inc1;

					if ((x_pixel ==
					     x0) &
					    (y_pixel == y0) & (z_pixel == z0)) {

						value += scale_factor * x[ray];
						break;

					} else if (((inc1 ==
					             -1) & (*pixel1 <= max)) |
					           ((inc1 ==
					             1) & (*pixel1 >= max)) ) {

						break;
					}

				}

			}

		}
	}

	if (value != 0)
		y[wrIdx] = value;

}
#endif

void dyn_calc_kernel_wrapper( dim3 blocks, dim3 threads, size_t dynMem,
                              cudaStream_t stream,
                              cublasOperation_t transA, int sub_vol,
                              int num_emitters, int num_receivers, int rv_x,
                              int rv_y, int rv_z,
                              float scale_factor, const int *x_receivers,
                              const int *y_receivers, const int *z_receivers,
                              const int *x_emitters,
                              const int *y_emitters, const int *z_emitters,
                              const float *x, float *y,
                              const unsigned int* use_path,
                              textureParams<float> *params,
                              const unsigned int emitter0) {
#ifdef PROFILING0

	if (transA == CUBLAS_OP_N) {
		profile_info[30].valid = true;
		sprintf(profile_info[30].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 4 );
	} else {
		profile_info[31].valid = true;
		sprintf(profile_info[31].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 7 );
	}
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif

	cudaMemcpy3DParms copyParams = {0};

	if (transA == CUBLAS_OP_N) {

		copyParams.srcPtr = params->pitchedDevPtr;
		copyParams.srcPos = make_cudaPos(0, 0, 0);
		copyParams.dstArray = params->inputArray;
		copyParams.dstPos = make_cudaPos(0, 0, 0);
		copyParams.extent = make_cudaExtent(rv_y, rv_x, rv_z);
		copyParams.kind = cudaMemcpyDeviceToDevice;

		HANDLE_ERROR(cudaMemcpy3DAsync( &(copyParams), stream));

		texInput.addressMode[0] = cudaAddressModeClamp;
		texInput.addressMode[1] = cudaAddressModeClamp;
		texInput.addressMode[2] = cudaAddressModeClamp;
		texInput.filterMode = cudaFilterModePoint;
		texInput.normalized = 0;

		HANDLE_ERROR(cudaBindTextureToArray(&texInput,
		                                    params->inputArray,
		                                    &(params->textDesc)));

		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel<  8,
		                                                      float>,
		                                    cudaFuncCachePreferL1  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 16,
		                                                     float>,
		                                    cudaFuncCachePreferL1  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 32,
		                                                     float>,
		                                    cudaFuncCachePreferL1  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 64,
		                                                     float>,
		                                    cudaFuncCachePreferL1  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 96,
		                                                     float>,
		                                    cudaFuncCachePreferL1  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel<128,
		                                                    float>,
		                                    cudaFuncCachePreferL1  ));

	} else { // surface memory for T-multiply --> does not improve performance!

		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel<  8,
		                                                      float>,
		                                    cudaFuncCachePreferShared  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 16,
		                                                     float>,
		                                    cudaFuncCachePreferShared  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 32,
		                                                     float>,
		                                    cudaFuncCachePreferShared  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 64,
		                                                     float>,
		                                    cudaFuncCachePreferShared  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 96,
		                                                     float>,
		                                    cudaFuncCachePreferShared  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel<128,
		                                                    float>,
		                                    cudaFuncCachePreferShared  ));

	}

	switch (sub_vol) {
	case   0:      dyn_calc_kernel<  0,
		                         float><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case   1:      dyn_calc_kernel<  1,
		                         float><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case   8:      dyn_calc_kernel<  8,
		                         float><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case  16:      dyn_calc_kernel< 16,
		                        float><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case  32:      dyn_calc_kernel< 32,
		                        float><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case  64:      dyn_calc_kernel< 64,
		                        float><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case  96:      dyn_calc_kernel< 96,
		                        float><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case 128:      dyn_calc_kernel<128,
		                       float><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	}
	HANDLE_ERROR(cudaGetLastError());

	if ( transA == CUBLAS_OP_N ) {
		HANDLE_ERROR(cudaUnbindTexture(&texInput));
	}

#ifdef PROFILING0
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime = 0;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));

	if (transA == CUBLAS_OP_N) {
		profile_info[30].time += elapsedTime;
		profile_info[30].runs++;
	} else {
		profile_info[31].time += elapsedTime;
		profile_info[31].runs++;
	}
#endif
}

void dyn_calc_kernel_wrapper( dim3 blocks, dim3 threads, size_t dynMem,
                              cudaStream_t stream,
                              cublasOperation_t transA, int sub_vol,
                              int num_emitters, int num_receivers, int rv_x,
                              int rv_y, int rv_z,
                              double scale_factor, const int *x_receivers,
                              const int *y_receivers, const int *z_receivers,
                              const int *x_emitters,
                              const int *y_emitters, const int *z_emitters,
                              const double *x, double *y,
                              const unsigned int* use_path,
                              textureParams<double> *params,
                              const unsigned int emitter0) {
#ifdef PROFILING0

	if (transA == CUBLAS_OP_N) {
		profile_info[30].valid = true;
		sprintf(profile_info[30].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 4 );
	} else {
		profile_info[31].valid = true;
		sprintf(profile_info[31].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 7 );
	}
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif

	if (transA == CUBLAS_OP_N) {

		cudaMemcpy3DParms copyParams = {0};

		copyParams.srcPtr = params->pitchedDevPtr;
		copyParams.srcPos = make_cudaPos(0, 0, 0);
		copyParams.dstArray = params->inputArray;
		copyParams.dstPos = make_cudaPos(0, 0, 0);
		copyParams.extent = make_cudaExtent(rv_y, rv_x, rv_z);
		copyParams.kind = cudaMemcpyDeviceToDevice;

		HANDLE_ERROR(cudaMemcpy3DAsync( &(copyParams), stream));

		texInput.addressMode[0] = cudaAddressModeClamp;
		texInput.addressMode[1] = cudaAddressModeClamp;
		texInput.addressMode[2] = cudaAddressModeClamp;
		texInput.filterMode = cudaFilterModePoint;
		texInput.normalized = 0;

		HANDLE_ERROR(cudaBindTextureToArray(&texInput0,
		                                    params->inputArray,
		                                    &(params->textDesc)));

		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel<  8,
		                                                      double>,
		                                    cudaFuncCachePreferL1  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 16,
		                                                     double>,
		                                    cudaFuncCachePreferL1  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 32,
		                                                     double>,
		                                    cudaFuncCachePreferL1  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 64,
		                                                     double>,
		                                    cudaFuncCachePreferL1  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 96,
		                                                     double>,
		                                    cudaFuncCachePreferL1  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel<128,
		                                                    double>,
		                                    cudaFuncCachePreferL1  ));

	} else {

		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel<  8,
		                                                      double>,
		                                    cudaFuncCachePreferShared  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 16,
		                                                     double>,
		                                    cudaFuncCachePreferShared  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 32,
		                                                     double>,
		                                    cudaFuncCachePreferShared  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 64,
		                                                     double>,
		                                    cudaFuncCachePreferShared  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel< 96,
		                                                     double>,
		                                    cudaFuncCachePreferShared  ));
		HANDLE_ERROR(cudaFuncSetCacheConfig(dyn_calc_kernel<128,
		                                                    double>,
		                                    cudaFuncCachePreferShared  ));
	}

	switch (sub_vol) {
	case   0:      dyn_calc_kernel<  0,
		                         double><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case   1:      dyn_calc_kernel<  1,
		                         double><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case   8:      dyn_calc_kernel<  8,
		                         double><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case  16:      dyn_calc_kernel< 16,
		                        double><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case  32:      dyn_calc_kernel< 32,
		                        double><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case  64:      dyn_calc_kernel< 64,
		                        double><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case  96:      dyn_calc_kernel< 96,
		                        double><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	case 128:      dyn_calc_kernel<128,
		                       double><< < blocks, threads, dynMem,
		        stream >> >
		        (transA, num_emitters, num_receivers, rv_x, rv_y, rv_z,
		         scale_factor,
		         x_receivers, y_receivers, z_receivers, x_emitters,
		         y_emitters,
		         z_emitters, x, y, use_path, emitter0); break;
	}
	HANDLE_ERROR(cudaGetLastError());

	if ( transA == CUBLAS_OP_N ) {
		HANDLE_ERROR(cudaUnbindTexture(&texInput0));
	}

#ifdef PROFILING0
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));

	if (transA == CUBLAS_OP_N) {
		profile_info[30].time += elapsedTime;
		profile_info[30].runs++;
	} else {
		profile_info[31].time += elapsedTime;
		profile_info[31].runs++;
	}
#endif
}

void sum_vol_up_kernel_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                               cudaStream_t stream,
                               int sub_vol, int dim_x, int dim_y, int dim_z,
                               double *y, const double *y_local) {
#ifdef PROFILING0
	profile_info[32].valid = true;
	sprintf(profile_info[32].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif
	switch (sub_vol) {
	case   8: sum_vol_up_kernel<  8, double,
		                      double2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	case  16: sum_vol_up_kernel< 16, double,
		                     double2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	case  32: sum_vol_up_kernel< 32, double,
		                     double2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	case  64: sum_vol_up_kernel< 64, double,
		                     double2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	case  96: sum_vol_up_kernel< 96, double,
		                     double2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	case 128: sum_vol_up_kernel<128, double,
		                    double2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	}
	HANDLE_ERROR(cudaGetLastError());
#ifdef PROFILING0
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[32].time += elapsedTime;
	profile_info[32].runs++;
#endif
}

void sum_vol_up_kernel_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                               cudaStream_t stream,
                               int sub_vol, int dim_x, int dim_y, int dim_z,
                               float *y, const float * y_local) {
#ifdef PROFILING0
	profile_info[32].valid = true;
	sprintf(profile_info[32].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif

	switch (sub_vol) {
	case   8: sum_vol_up_kernel<  8, float,
		                      float2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	case  16: sum_vol_up_kernel< 16, float,
		                     float2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	case  32: sum_vol_up_kernel< 32, float,
		                     float2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	case  64: sum_vol_up_kernel< 64, float,
		                     float2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	case  96: sum_vol_up_kernel< 96, float,
		                     float2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	case 128: sum_vol_up_kernel<128, float,
		                    float2><< < blocks, threads, dynMem,
		        stream >> >
		        (dim_x, dim_y, dim_z, y, y_local); break;
	}

	HANDLE_ERROR(cudaGetLastError());
#ifdef PROFILING0
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[32].time += elapsedTime;
	profile_info[32].runs++;
#endif
}

void sum_vol_up_kernel_inter_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                                     cudaStream_t stream,
                                     int sub_vol, bool accResult, int dim_x,
                                     int dim_y, int dim_z, double *y,
                                     const double *y_local) {
#ifdef PROFILING
	profile_info[33].valid = true;
	sprintf(profile_info[33].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif

	int pow = 256 / threads.x;

	while ( pow > sub_vol)
		pow >>= 1;

	threads.x *= pow;

	if ( accResult == true ) {

		switch (pow) {
		case   1: sum_vol_up_kernel_inter< 1, true,
			                           double><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   2: sum_vol_up_kernel_inter< 2, true,
			                           double><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   4: sum_vol_up_kernel_inter< 4, true,
			                           double><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   8: sum_vol_up_kernel_inter< 8, true,
			                           double><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case  16: sum_vol_up_kernel_inter<16, true,
			                          double><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		}

	} else {

		switch (pow) {
		case   1: sum_vol_up_kernel_inter< 1, false,
			                           double><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   2: sum_vol_up_kernel_inter< 2, false,
			                           double><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   4: sum_vol_up_kernel_inter< 4, false,
			                           double><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   8: sum_vol_up_kernel_inter< 8, false,
			                           double><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case  16: sum_vol_up_kernel_inter<16, false,
			                          double><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		}
	}
	HANDLE_ERROR(cudaGetLastError());
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[33].time += elapsedTime;
	profile_info[33].runs++;
#endif
}

void sum_vol_up_kernel_inter_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                                     cudaStream_t stream,
                                     int sub_vol, bool accResult, int dim_x,
                                     int dim_y, int dim_z, float *y,
                                     const float * y_local) {
#ifdef PROFILING
	profile_info[34].valid = true;
	sprintf(profile_info[34].name, "%s\t (%s,%i)\0", __FUNCTION__, __FILE__,
	        __LINE__ - 4 );
	HANDLE_ERROR(cudaEventRecord(start_cu));
#endif

	int pow = 256 / threads.x;

	while ( pow > sub_vol)
		pow >>= 1;

	threads.x *= pow;

	if ( accResult == true ) {

		switch (pow) {
		case   1: sum_vol_up_kernel_inter< 1, true,
			                           float><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   2: sum_vol_up_kernel_inter< 2, true,
			                           float><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   4: sum_vol_up_kernel_inter< 4, true,
			                           float><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   8: sum_vol_up_kernel_inter< 8, true,
			                           float><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case  16: sum_vol_up_kernel_inter<16, true,
			                          float><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		}
	} else {

		switch (pow) {
		case   1: sum_vol_up_kernel_inter< 1, false,
			                           float><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   2: sum_vol_up_kernel_inter< 2, false,
			                           float><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   4: sum_vol_up_kernel_inter< 4, false,
			                           float><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case   8: sum_vol_up_kernel_inter< 8, false,
			                           float><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		case  16: sum_vol_up_kernel_inter<16, false,
			                          float><< < blocks, threads,
			        dynMem * pow, stream >> >
			        (sub_vol, dim_x, dim_y, dim_z, y, y_local);
			break;
		}
	}
	HANDLE_ERROR(cudaGetLastError());
#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop_cu));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop_cu));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_cu, stop_cu));
	profile_info[34].time += elapsedTime;
	profile_info[34].runs++;
#endif
}
