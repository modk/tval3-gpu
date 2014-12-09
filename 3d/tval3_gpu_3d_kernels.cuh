#ifndef TVAL3_GPU_KERHELS_CUH_
#define TVAL3_GPU_KERHELS_CUH_

#include <stdio.h>

template<class Type>
struct textureParams {
	size_t subVol;

	cudaChannelFormatDesc textDesc;
	cudaArray *inputArray;
	cudaPitchedPtr pitchedDevPtr;

	textureParams(int dim_x, int dim_y, int dim_z, int sub_vol =
	                      1) : subVol(sub_vol) {

		// texture input
		HANDLE_ERROR(cudaMalloc3D(&pitchedDevPtr, 
					make_cudaExtent(dim_y * sizeof(Type), dim_x, dim_z)));

		if ( sizeof(Type) == 4)
			textDesc = cudaCreateChannelDesc(32, 0, 0, 0, 
					cudaChannelFormatKindFloat);

		else
			textDesc = cudaCreateChannelDesc(32, 32, 0, 0, 
					cudaChannelFormatKindFloat);


		HANDLE_ERROR(cudaMalloc3DArray( &inputArray, &textDesc,
					make_cudaExtent(dim_y * sizeof(Type), dim_x, dim_z), 0));

	}

	~textureParams() {
		//HANDLE_ERROR(cudaFreeArray(inputArray));
		//HANDLE_ERROR(cudaFree(pitchedP));
	}
};

//-----------------------------------------------------------------------------
// single precision (float)
//-----------------------------------------------------------------------------
void vec_add_kernel_wrapper(int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, float *result, const float alpha,
                            const float *x, const float *y, const float *z);

void D_kernel_no_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                             cudaStream_t stream,
                             const float *U, float *Ux, float *Uy, float *Uz,
                             int dim_x, int dim_y, int dim_z);

void D_kernel_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                          cudaStream_t stream, size_t texSize,
                          const float *U, float *Ux, float *Uy, float *Uz,
                          int dim_x, int dim_y, int dim_z);

void Dt_kernel_no_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                              cudaStream_t stream,
                              const float *X, const float *Y, const float *Z,
                              float *res, int dim_x, int dim_y, int dim_z);

void Dt_kernel_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                           cudaStream_t stream, size_t texSizeX,
                           size_t texSizeY, size_t texSizeZ,
                           const float *X, const float *Y, const float *Z,
                           float *res, int dim_x, int dim_y, int dim_z);

void lam2_4_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, const float *Vx, const float *Vy,
                            const float *Vz, const float *sigmaX,
                            const float *sigmaY, const float *sigmaZ,
                            float *tmp_lam2, float *tmp_lam4);

void lam3_5_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, const float *Au, const float *b,
                            const float *delta, float *tmp_lam3,
                            float *tmp_lam5);

void get_tau_kernel_wrapper( int blocks, int threads, size_t dynMem,
                             cudaStream_t stream,
                             int N, const float *g, const float *gp,
                             const float *g2, const float *g2p,
                             const float *uup,
                             float muDbeta, float *dg, float *dg2,
                             float *tmp_ss, float *tmp_sy);

void nonneg_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, float *vec);

void shrinkage_kernel_wrapper( int blocks, int threads, size_t dynMem,
                               cudaStream_t stream,
                               int N, const float *Ux, const float *Uy,
                               const float *Uz, const float *sigmaX,
                               const float *sigmaY, const float *sigmaZ,
                               float beta, float *Wx, float *Wy, float *Wz,
                               float *tmp);

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
                              textureParams<float> *params = NULL,
                              const unsigned int emitter0 = 0);

void sum_vol_up_kernel_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                               cudaStream_t stream,
                               int sub_vol, int dim_x, int dim_y, int dim_z,
                               float *y, const float * y_local);

void sum_vol_up_kernel_inter_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                                     cudaStream_t stream,
                                     int sub_vol, bool accResult, int dim_x,
                                     int dim_y, int dim_z, float *y,
                                     const float * y_local);

//-----------------------------------------------------------------------------
// double precision (double)
//-----------------------------------------------------------------------------
void vec_add_kernel_wrapper(int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, double *result, const double alpha,
                            const double *x, const double *y, const double *z);

void D_kernel_no_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                             cudaStream_t stream,
                             const double *U, double *Ux, double *Uy,
                             double *Uz, int dim_x, int dim_y, int dim_z);

void D_kernel_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                          cudaStream_t stream, size_t texSize,
                          const double *U, double *Ux, double *Uy, double *Uz,
                          int dim_x, int dim_y, int dim_z);

void Dt_kernel_no_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                              cudaStream_t stream,
                              const double *X, const double *Y, const double *Z,
                              double *res, int dim_x, int dim_y, int dim_z);

void Dt_kernel_tex_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                           cudaStream_t stream, size_t texSizeX,
                           size_t texSizeY, size_t texSizeZ,
                           const double *X, const double *Y, const double *Z,
                           double *res, int dim_x, int dim_y, int dim_z);

void lam2_4_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, const double *Vx, const double *Vy,
                            const double *Vz, const double *sigmaX,
                            const double *sigmaY, const double *sigmaZ,
                            double *tmp_lam2, double *tmp_lam4);

void lam3_5_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, const double *Au, const double *b,
                            const double *delta, double *tmp_lam3,
                            double *tmp_lam5);

void get_tau_kernel_wrapper( int blocks, int threads, size_t dynMem,
                             cudaStream_t stream,
                             int N, const double *g, const double *gp,
                             const double *g2, const double *g2p,
                             const double *uup,
                             double muDbeta, double *dg, double *dg2,
                             double *tmp_ss, double *tmp_sy);

void nonneg_kernel_wrapper( int blocks, int threads, size_t dynMem,
                            cudaStream_t stream,
                            int N, double *vec);

void shrinkage_kernel_wrapper( int blocks, int threads, size_t dynMem,
                               cudaStream_t stream,
                               int N, const double *Ux, const double *Uy,
                               const double *Uz, const double *sigmaX,
                               const double *sigmaY, const double *sigmaZ,
                               double beta, double *Wx, double *Wy, double *Wz,
                               double *tmp);

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
                              textureParams<double> *params = NULL,
                              const unsigned int emitter0 = 0);

void sum_vol_up_kernel_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                               cudaStream_t stream,
                               int sub_vol, int dim_x, int dim_y, int dim_z,
                               double *y, const double * y_local);

void sum_vol_up_kernel_inter_wrapper(dim3 blocks, dim3 threads, size_t dynMem,
                                     cudaStream_t stream,
                                     int sub_vol, bool accResult, int dim_x,
                                     int dim_y, int dim_z, double *y,
                                     const double * y_local);

//-----------------------------------------------------------------------------
// conversion functions: D2S and S2D
//-----------------------------------------------------------------------------
void convS2D_GPU_kernel_wrapper(int blocks, int threads, size_t dynMem,
                                cudaStream_t stream,
                                int N, double *output, float *input);
void convD2S_GPU_kernel_wrapper(int blocks, int threads, size_t dynMem,
                                cudaStream_t stream,
                                int N, float *output, double *input);

#endif
