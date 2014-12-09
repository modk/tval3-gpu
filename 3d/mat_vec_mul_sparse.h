#ifndef MAT_VEC_MUL_SPARSE_H_
#define MAT_VEC_MUL_SPARSE_H_

#include "profile_gpu.h"
#include "handle_error.h"
#include "time.h"
#include "sys/time.h"

#include <stdio.h>
#include "tval3_gpu_3d_kernels.cuh"

/******************************************************************************
* Multi-GPU processing for sparse matrices (single-threaded initialization)
******************************************************************************/
#if (1)
//-----------------------------------------------------------------------------
// type definition for sparse matrices distributed over multiple GPUS
//-----------------------------------------------------------------------------
template<class Type>
class sparse_mm_multGPU {
	public:
		int mainDevice;
		int numDevices;
		bool cscVertical;
		bool mainUsedForSpMV;
		int *deviceArray;
		int numCSC;
		int *cscColIdx;
		sparse_mat_device<Type> **A_csc;
		int *csrRowIdx;
		int numCSR;
		sparse_mat_device<Type> **A_csr;
		Type **memA;
		Type **memB;
		Type *memAcc;
		cudaStream_t *streams;
		cusparseHandle_t *cs_handle;
		cusparseMatDescr_t *descrA;
		const bool allocedInConstructor;

		sparse_mm_multGPU(const sparse_mat_host<Type> &A, int numReqDevices,
				int *reqDeviceArray, int mainDevice,
				bool verticalCSCCut = false);
		sparse_mm_multGPU(int mainDevice, int numDevices, bool cscVertical,
				bool mainUsedForSpMV, int *deviceArray,
				int numCSC, int *cscColIdx, sparse_mat_device<Type> **A_csc,
				int *csrRowIdx, int numCSR, sparse_mat_device<Type> **A_csr,
				Type **x_local, Type **y_local, Type *memTmp,
				cudaStream_t *streams, cusparseHandle_t *cs_handle,
				cusparseMatDescr_t *descrA );
		~sparse_mm_multGPU();

};

template<class Type>
sparse_mm_multGPU<Type>::sparse_mm_multGPU(int mainDevice, int numDevices,
                                           bool cscVertical,
                                           bool mainUsedForSpMV,
                                           int *deviceArray,
                                           int numCSC, int *cscColIdx,
                                           sparse_mat_device<Type> **A_csc,
                                           int *csrRowIdx, int numCSR,
                                           sparse_mat_device<Type> **A_csr,
                                           Type **x_local, Type **y_local,
                                           Type *memTmp, cudaStream_t *streams,
                                           cusparseHandle_t *cs_handle,
                                           cusparseMatDescr_t *descrA ) :

	mainDevice(mainDevice), numDevices(numDevices),
	cscVertical(cscVertical), mainUsedForSpMV(mainUsedForSpMV), deviceArray(
	        deviceArray),
	numCSC(numCSC), cscColIdx(cscColIdx), A_csc(A_csc),
	csrRowIdx(csrRowIdx), numCSR(numCSR), A_csr(A_csr),
	memA(x_local), memB(y_local), memAcc(memTmp), streams(streams),
	cs_handle(cs_handle), descrA(descrA), allocedInConstructor(false) {
};


template<class Type>
sparse_mm_multGPU<Type>::sparse_mm_multGPU(const sparse_mat_host<Type> &A,
                                           int numReqDevices,
                                           int *reqDeviceArray, int mainDevice,
                                           bool verticalCSCCut ) :
	mainDevice(mainDevice), numCSC(0), numCSR(0),
	allocedInConstructor(true) {

	double difftime;
	timeval start, stop;

	printf("---------------------------------- sparse init ----------------------------------\n");

	mainUsedForSpMV = false;

	// Get names number and names of GPU devices --------------------------------
	gettimeofday(&start, NULL);

	int numAvailDevices = 0;
	cudaGetDeviceCount(&numAvailDevices);
	struct cudaDeviceProp *devices = (struct cudaDeviceProp*)malloc(
	        numAvailDevices * sizeof(struct cudaDeviceProp) );

	int *canAccessPeer = new int[numAvailDevices];

	printf("Detected %i CUDA capable devices: ", numAvailDevices);

	for (int k = 0; k < numAvailDevices; k++ ) {

		cudaGetDeviceProperties( &devices[k], k);
		printf("%s ", devices[k].name);

		cudaDeviceCanAccessPeer(&(canAccessPeer[k]), k, mainDevice);

	}

	printf("\n");

	if ( numAvailDevices < numReqDevices) {
		printf( "WARNING: number of chosen GPUs exceeds maximum number! Reducing request to %i devices\n",
		        numAvailDevices);
		numReqDevices = numAvailDevices;
	} else if ( numAvailDevices > numReqDevices) {
		printf(
		        "INFO: number of requested GPUs is smaller than number available\n",
		        numReqDevices);
	}

	deviceArray = new int[numReqDevices];

	printf("Setting %s (device %i) as main device!\n",
	       devices[mainDevice].name, mainDevice);

	numDevices = 0;
	for (int i = 0; i < numReqDevices; i++) {

		if ((canAccessPeer[reqDeviceArray[i]] ==
		     1) | (reqDeviceArray[i] == mainDevice)) {
			deviceArray[numDevices] = reqDeviceArray[i];
			numDevices++;
		} else {
			printf(
			        "Device %s can not be used as it can not access peer memory on main device!\n",
			        devices[reqDeviceArray[i]].name);
		}

		if ( reqDeviceArray[i] == mainDevice )
			mainUsedForSpMV = true;

	}

	if ( numDevices < numReqDevices ) {
		printf("--> Actually processing on only %i devices, namely:",
		       numDevices);
		for (int i = 0; i < numDevices; i++)
			printf(" %s", devices[deviceArray[i]].name);
		printf("\n");
	} else {
		printf("--> Actually processing on %i devices, namely:",
		       numDevices);
		for (int i = 0; i < numDevices; i++)
			printf(" %s", devices[deviceArray[i]].name);
		printf("\n");
	}

	// Determine optimal number of NNZ per submatrix ----------------------------
	cscVertical = verticalCSCCut;

	int nnzMainOpt = 0, nnzOpt = 0;

	if ((mainUsedForSpMV == true) & (numDevices > 1)) {

		int currDevice;
		HANDLE_ERROR(cudaGetDevice(&currDevice));
		size_t totalMem, freeMem;
		HANDLE_ERROR(cudaSetDevice(mainDevice));
		HANDLE_ERROR(cudaMemGetInfo( &freeMem, &totalMem ));
		HANDLE_ERROR(cudaSetDevice(currDevice));

		// subtract memory needed for further matrices on main GPU (numbers are 
		// constant, only dependent on resolution/signals)
		freeMem -= (28 + 4 + 3) * A.dim_x * sizeof(Type) +
			(7 + 1 + 1) * A.dim_y * sizeof(Type);

		// subtract additional memory needed for accumulation of partial volumes if 
		// horizontal CSC slices are distributed
		if (cscVertical == false)
			freeMem -= A.dim_x * (numDevices - 1) * sizeof(Type);

		// determine maximum number of NNZs on main device: distinguish if CSR, CSC
		// or both matrices are on the device
		if (A.format == sparse_mat_both)
			nnzMainOpt = floor( (double)(freeMem - (A.dim_y * sizeof(int)) -
						(A.dim_x * sizeof(int))) / 
					(double)(2 * (sizeof(Type) + sizeof(int))) );

		else if (A.format == sparse_mat_csr)
			nnzMainOpt = floor( (double)(freeMem - (A.dim_y * sizeof(int))) /
			               (double)(sizeof(Type) + sizeof(int)) );

		else
			nnzMainOpt = floor( (double)(freeMem - (A.dim_x * sizeof(int))) /
			               (double)(sizeof(Type) + sizeof(int)) );

		nnzOpt = ceil((double)(A.nnz - nnzMainOpt) / (double)(numDevices - 1));

		// check if nnzMainOpt is by far too large (due to small matrix size or 
		// large free GPU memory)
		double mainOverFactor = 1; //1.25; // must be < numDevices

		if (nnzMainOpt > A.nnz) { //complete matrix would fit onto device

			nnzMainOpt = floor((double)(A.nnz) / (double)(numDevices + 
						mainOverFactor - 1) * mainOverFactor);
			nnzOpt = ceil ((double)(A.nnz - nnzMainOpt) / (double)(numDevices - 1));

		} else if ( (double)nnzMainOpt / mainOverFactor > (double)nnzOpt ) { 
			// if mainOverFactor more NNZs

			nnzMainOpt = floor((double)(A.nnz) / (double)(numDevices + 
						mainOverFactor - 1) * mainOverFactor);
			nnzOpt = ceil ((double)(A.nnz - nnzMainOpt) / (double)(numDevices - 1));

		} else if (nnzMainOpt < nnzOpt) {
			printf("Reduced NNZs on main device due to lack of memory: " \ "
					nnzMainOpt=%i, nnzOpt=%i, nnzMainOpt/nnzOpt=%.3f\n", nnzMainOpt, 
					nnzOpt, (double)nnzMainOpt / (double)nnzOpt);
		}

	} else {

		nnzMainOpt = ceil((double)(A.nnz) / (double)(numDevices));
		nnzOpt = nnzMainOpt;

	}

	gettimeofday(&stop, NULL);
	difftime = (double)((stop.tv_sec * 1000000 + stop.tv_usec) -
			(start.tv_sec * 1000000 + start.tv_usec));
	//printf("time 1: %.3fms\n", difftime / 1000);

	// Generate data elements for sub-matrices  ---------------------------------
	gettimeofday(&start, NULL);

	int *nnzCSRAct, *nnzCSCAct;
	int **buff_csr_ptr, **buff_csc_ptr;

	int **buff_csr_ind, **buff_csc_ind;
	Type **buff_csr_val, **buff_csc_val;

	int *csr_dim_x, *csr_dim_y, *csc_dim_x, *csc_dim_y;

	int currCSRidx = 0;
	int currCSRRowIdx = 0;
	int currCSRRowVal = 0;

	int currCSCidx = 0;
	int currCSCColIdx = 0;
	int currCSCColVal = 0;

	if ( (A.format == sparse_mat_csr) || (A.format == sparse_mat_both)) {

		csrRowIdx = new int[numDevices];
		nnzCSRAct = new int[numDevices];
		buff_csr_ptr = new int*[numDevices];
		buff_csr_ind = new int*[numDevices];
		buff_csr_val = new Type*[numDevices];
		csr_dim_x = new int[numDevices];
		csr_dim_y = new int[numDevices];

		currCSRRowVal = A.csr_ptr()[currCSRRowIdx];

		numCSR = numDevices;

	}

	if ( (A.format == sparse_mat_csc) || (A.format == sparse_mat_both)) {

		cscColIdx = new int[numDevices];
		nnzCSCAct = new int[numDevices];
		buff_csc_ptr = new int*[numDevices];
		buff_csc_ind = new int *[numDevices];
		buff_csc_val = new Type*[numDevices];
		csc_dim_x = new int[numDevices];
		csc_dim_y = new int[numDevices];

		currCSCColVal = A.csc_ptr()[currCSCColIdx];

		numCSC = numDevices;
	}

	// Generate and populate CSR sub-matrix until final block has been reached
	for (int i = 0; i < numCSR; i++) {


		// matrix cut and distributed in y dimension: x dimension always full
		csr_dim_x[i] = A.dim_x; 
		if (deviceArray[i] != mainDevice )
			nnzCSRAct[i] = nnzOpt;
		else
			nnzCSRAct[i] = nnzMainOpt;

		csrRowIdx[i] = currCSRRowIdx;

		buff_csr_ind[i] = (int*)A.csr_ind() + currCSRidx;
		buff_csr_val[i] = (Type*)A.csr_val() + currCSRidx;
		buff_csr_ptr[i] = new int[A.dim_y + 1];

		if ( currCSRidx + nnzCSRAct[i] >= A.nnz ) {

			// end of matrix reached with this block, make this block fitting and 
			// reduce numGPUs if whole device unused!
			
			//do not use further GPUs (this should hopefully never happen)
			numCSR = i + 1;       
			int k = 0;
			while ( currCSRRowIdx + k <= A.dim_y) {

				buff_csr_ptr[i][k] = A.csr_ptr()[currCSRRowIdx + k] - currCSRRowVal;
				k++;
			}

			k--; // last increment is one to many!

			nnzCSRAct[i] = A.csr_ptr()[currCSRRowIdx + k] - currCSRidx;
			csr_dim_y[i] = k;

		} else {

			int k = 0;
			while ((A.csr_ptr()[currCSRRowIdx + k] < currCSRidx + nnzCSRAct[i]) &
			        (currCSRRowIdx + k < A.dim_y)) {

				buff_csr_ptr[i][k] = A.csr_ptr()[currCSRRowIdx + k] - currCSRRowVal;
				k++;
			}

			buff_csr_ptr[i][k] = A.csr_ptr()[currCSRRowIdx + k] - currCSRRowVal;

			nnzCSRAct[i] = A.csr_ptr()[currCSRRowIdx + k] - currCSRidx;
			csr_dim_y[i] = k;

			// set values for next iteration
			currCSRRowIdx += k;
			currCSRRowVal = A.csr_ptr()[currCSRRowIdx];
			currCSRidx += nnzCSRAct[i];
		}
	}

	// Generate and populate CSC sub-matrix until final block has been reached: 
	// depends if horizontally or vertically cut
	if ( cscVertical == true ) {

		// traditional vertical cut (along signal axis Y): approx. equal number of 
		// NNZs, but different number of columns per submatrix
		for (int i = 0; i < numCSC; i++) {

			// matrix cut and distributed in x dimension: y dimension always full
			csc_dim_y[i] = A.dim_y; 

			if (deviceArray[i] != mainDevice )
				nnzCSCAct[i] = nnzOpt;
			else
				nnzCSCAct[i] = nnzMainOpt;

			cscColIdx[i] = currCSCColIdx;

			// Generate and populate CSC sub-matrix
			buff_csc_ind[i] = (int*)A.csc_ind() + currCSCidx;
			buff_csc_val[i] = (Type*)A.csc_val() + currCSCidx;
			buff_csc_ptr[i] = new int[A.dim_x + 1];

			if ( currCSCidx + nnzCSCAct[i] >= A.nnz ) { 
				// end of matrix reached with this block, make this block fitting and 
				// reduce numGPUs if whole device unused!

				//end of matrix is reached, do not use further GPUs (this should 
				//hopefully never happen)
				numCSC = i + 1;       

				int k = 0;
				while (currCSCColIdx + k <= A.dim_x) {

					buff_csc_ptr[i][k] =
					        A.csc_ptr()[currCSCColIdx +
					                    k] - currCSCColVal;
					k++;
				}

				k--; // last increment is one to many!

				nnzCSCAct[i] =
				        A.csc_ptr()[currCSCColIdx +
				                    k] - currCSCidx;
				csc_dim_x[i] = k;

			} else {

				int k = 0;
				while ((A.csc_ptr()[currCSCColIdx + k] < currCSCidx + nnzCSCAct[i]) & 
						(currCSCColIdx + k <= A.dim_x) ) {

					buff_csc_ptr[i][k] = A.csc_ptr()[currCSCColIdx + k] - currCSCColVal;
					k++;
				}

				buff_csc_ptr[i][k] = A.csc_ptr()[currCSCColIdx + k] - currCSCColVal;

				nnzCSCAct[i] = A.csc_ptr()[currCSCColIdx + k] - currCSCidx;
				csc_dim_x[i] = k;

				// set values for next iteration
				currCSCColIdx += k;
				currCSCColVal = A.csc_ptr()[currCSCColIdx];
				currCSCidx += nnzCSCAct[i];
			}
		}

	} else {

		// innovative horizontal cut (along volume axis, X): approx. equal number 
		// of NNZs and rows per submatrix
		// needs accumulation for T-multiply
		numCSC = numCSR;

		if (numCSC > 1) {

			int *currIdx = new int[numCSC];
			matDim *dimY = new matDim[numCSC];

			dimY[0].min = 0;

			for (int i = 0; i < numCSC; i++) {

				nnzCSCAct[i] = nnzCSRAct[i];
				csc_dim_x[i] = csr_dim_x[i];
				csc_dim_y[i] = csr_dim_y[i];

				buff_csc_ptr[i] = new int [A.dim_x + 1];
				buff_csc_ind[i] = new int [nnzCSCAct[i]];
				buff_csc_val[i] = new Type[nnzCSCAct[i]];

				cscColIdx[i] = csrRowIdx[i];

				dimY[i].max = dimY[i].min + csc_dim_y[i];

				if (i < numCSC -
				    1) dimY[i + 1].min = dimY[i].max;

				currIdx[i] = 0;
				buff_csc_ptr[i][0] = 0;

			}

			int currRow = 0, currDevice = 0, currCol = 0;

			int i = 0;
			while (i < A.nnz) {

				currRow = A.csc_ind()[i];

				if ( (currRow <
				      dimY[currDevice].max) &
				     (i < A.csc_ptr()[currCol + 1]) ) {                                  
						 // simply add element to matrix and go on

					buff_csc_ind[currDevice][currIdx[currDevice]] = currRow - 
						dimY[currDevice].min;
					buff_csc_val[currDevice][currIdx[currDevice]] = A.csc_val()[i];
					currIdx[currDevice]++;

					i++;

				} else { 
					// slice of current device passed, try next one or go to next column

					// finalize column for this device
					buff_csc_ptr[currDevice][currCol + 1] = currIdx[currDevice];

					if ( currDevice < numCSC - 1) {
						currDevice++;
					} else {
						currDevice = 0;
						currCol++;
					}
				}
			}

			// finalize very last column for last device
			while ( currCol < csc_dim_x[currDevice] ) {

				buff_csc_ptr[currDevice][currCol + 1] = currIdx[currDevice];

				if ( currDevice < numCSC - 1) {
					currDevice++;
				} else {
					currDevice = 0;
					currCol++;
				}
			}

		} else {

			nnzCSCAct[0] = nnzCSRAct[0];
			csc_dim_x[0] = csr_dim_x[0];
			csc_dim_y[0] = csr_dim_y[0];

			buff_csc_ptr[0] = (int*)A.csc_ptr();
			buff_csc_ind[0] = (int*)A.csc_ind();
			buff_csc_val[0] = (Type*)A.csc_val();

			cscColIdx[0] = csrRowIdx[0];

		}

	}

	numDevices = max(numCSR, numCSC);

	gettimeofday(&stop, NULL);
	difftime = (double)((stop.tv_sec * 1000000 + stop.tv_usec) -
	                 (start.tv_sec * 1000000 + start.tv_usec));
	//printf("time 2: %.3fms\n", difftime / 1000);

	// Generate handles and device matrices, copy data to devices ---------------
	gettimeofday(&start, NULL);

	cs_handle = new cusparseHandle_t[numDevices];
	descrA = new cusparseMatDescr_t[numDevices];
	streams = new cudaStream_t[numDevices];

	A_csc = new sparse_mat_device<Type>*[numCSC];
	A_csr = new sparse_mat_device<Type>*[numCSR];

	memA = new Type*[numDevices];
	memB = new Type*[numDevices];

	printf("A: nnz=%i, dim_x=%i, dim_y=%i\n", A.nnz, A.dim_x, A.dim_y);

	for (int i = 0; i < numDevices; i++) {

		HANDLE_ERROR(cudaSetDevice(deviceArray[i]));

		if (deviceArray[i] != mainDevice )
			HANDLE_ERROR(cudaDeviceReset());

		int sizeA = 0;
		int sizeB = 0;

		int nnz = 0;

		if (deviceArray[i] != mainDevice )
			nnz = nnzOpt;
		else
			nnz = nnzMainOpt;

		if ( i < numCSR ) {

			A_csr[i] = new sparse_mat_device<Type>(csr_dim_y[i],
			                                       csr_dim_x[i],
			                                       nnzCSRAct[i],
			                                       buff_csr_val[i],
			                                       buff_csr_ptr[i],
			                                       buff_csr_ind[i],
			                                       sparse_mat_csr);

			delete[] buff_csr_ptr[i];

			sizeA = max(sizeA, A_csr[i]->dim_x);
			sizeB = max(sizeB, A_csr[i]->dim_y);

			printf("A_csr[%i]: nnz=%i (%+i, %2.1f\%), dim_x=%i (%2.1f\%), " \ "
					dim_y=%i (%2.1f\%)\t", deviceArray[i], A_csr[i]->nnz, 
					A_csr[i]->nnz - nnz, (A_csr[i]->nnz / (float)A.nnz) * 100, 
					A_csr[i]->dim_x, (A_csr[i]->dim_x / (float)A.dim_x) * 100,
					A_csr[i]->dim_y, (A_csr[i]->dim_y / (float)A.dim_y) * 100);
		}

		if ( i < numCSC ) {

			A_csc[i] = new sparse_mat_device<Type>(csc_dim_y[i],
			                                       csc_dim_x[i],
			                                       nnzCSCAct[i],
			                                       buff_csc_val[i],
			                                       buff_csc_ptr[i],
			                                       buff_csc_ind[i],
			                                       sparse_mat_csc);

			delete[] buff_csc_ptr[i];

			if ( cscVertical == true) {
				sizeA = max(sizeA, A_csc[i]->dim_y);
				sizeB = max(sizeB, A_csc[i]->dim_x);
			} else {
				sizeA = max(sizeA, A_csc[i]->dim_x);
				sizeB = max(sizeB, A_csc[i]->dim_y);
			}

			printf("A_csc[%i]: nnz=%i (%+i, %2.1f\%), dim_x=%i (%2.1f\%), " \ "
					dim_y=%i (%2.1f\%)\t", deviceArray[i], A_csc[i]->nnz, 
					A_csc[i]->nnz - nnz, (A_csc[i]->nnz / (float)A.nnz) * 100, 
					A_csc[i]->dim_x, (A_csc[i]->dim_x / (float)A.dim_x) * 100,
					A_csc[i]->dim_y, (A_csc[i]->dim_y / (float)A.dim_y) * 100);
		}

		// Initialize necessary cuSparse handles
		HANDLE_ERROR(cusparseCreate(&(cs_handle[i])));
		// general matrix, zero based indexing
		HANDLE_ERROR(cusparseCreateMatDescr(&(descrA[i]))); 

		HANDLE_ERROR(cudaStreamCreate(&(streams[i])));

		if ( deviceArray[i] != mainDevice ) {

			HANDLE_ERROR(cudaDeviceEnablePeerAccess(mainDevice, 0));

			HANDLE_ERROR(cudaMalloc((void**)&(memA[i]), sizeA *
			                        sizeof(Type)));
			HANDLE_ERROR(cudaMalloc((void**)&(memB[i]), sizeB *
			                        sizeof(Type)));
		}
	}

	HANDLE_ERROR(cudaSetDevice(mainDevice));

	if ((cscVertical == false) & (numCSC > 1) ) {

		if ( mainUsedForSpMV == true)
			HANDLE_ERROR(cudaMalloc((void**)(&memAcc),
			                        (int)A.dim_x * (numCSC - 1) * sizeof(Type)));
		else
			HANDLE_ERROR(cudaMalloc((void**)(&memAcc),
			                        (int)A.dim_x * numCSC * sizeof(Type)));

	} else { 
		// (numCSC == 1) --> only one device, no buffer needed. memTmp will be set
		// to result in mul-routine

		memAcc = NULL;
	}

	for (int i = 0; i < numDevices; i++) {

		size_t subMat_size = 0;  // size in bytes

		if ( i < numCSR ) {
		// CSR matrix
			subMat_size += (A_csr[i]->nnz * sizeof(Type)) + (A_csr[i]->nnz *
			         sizeof(int)) + (A_csr[i]->dim_y * sizeof(int));
		}

		if ( i < numCSC ) {
			// CSC matrix
			subMat_size += (A_csc[i]->nnz * sizeof(Type)) + (A_csc[i]->nnz *
			         sizeof(int)) + (A_csc[i]->dim_x * sizeof(int));
		}

		if (subMat_size < 1024) {
			printf("Mem size for sub matrices on GPU %i: %i Byte\n",
			       deviceArray[i], subMat_size );
		} else if (subMat_size < 1024 * 1024) {
			printf("Mem size for sub matrices on GPU %i: %i kByte\n",
			        deviceArray[i], subMat_size / 1024);
		} else if (subMat_size >= 1024 * 1024) {
			printf("Mem size for sub matrices on GPU %i: %i MByte\n",
			        deviceArray[i], subMat_size / (1024 * 1024));
		}

		printf(" ("); memcheck(deviceArray[i], false); printf(")\n");
	}

	gettimeofday(&stop, NULL);
	difftime = (double)((stop.tv_sec * 1000000 + stop.tv_usec) -
	                 (start.tv_sec * 1000000 + start.tv_usec));
	printf("Time for GPU init: %.3fms\n", difftime / 1000);

	// Finalize: clear intermediate arrays and set context (back to) main device 
	if (numCSR > 0) {
		delete[] nnzCSRAct;
		delete[] buff_csr_ptr;
		delete[] buff_csr_ind;
		delete[] buff_csr_val;
		delete[] csr_dim_x;
		delete[] csr_dim_y;
	}

	if (numCSC > 0) {
		delete[] nnzCSCAct;
		delete[] buff_csc_ptr;
		delete[] buff_csc_ind;
		delete[] buff_csc_val;
		delete[] csc_dim_x;
		delete[] csc_dim_y;
	}

}

template<class Type>
sparse_mm_multGPU<Type>::~sparse_mm_multGPU() {

	if ( allocedInConstructor == true) {

		for (int i = 0; i < numDevices; i++) {

			HANDLE_ERROR(cudaSetDevice(deviceArray[i]));

			HANDLE_ERROR(cusparseDestroyMatDescr(descrA[i]));
			HANDLE_ERROR(cusparseDestroy(cs_handle[i]));
			HANDLE_ERROR(cudaStreamDestroy(streams[i]));

			cudaFree((void*)memA[i]);
			cudaFree(memB[i]);

			if ( i < numCSR )
				delete(A_csc[i]);

			if ( i < numCSR )
				delete(A_csr[i]);

		}

		HANDLE_ERROR(cudaSetDevice(mainDevice));

		delete(descrA);
		delete(cs_handle);

		delete(A_csr);
		delete(A_csc);

		if (deviceArray != NULL) delete(deviceArray);
		deviceArray = NULL;
	}
}

//-----------------------------------------------------------------------------
// mat_vec_mul for sparse matrices on distributed over multiple GPUs
//-----------------------------------------------------------------------------
template<class Type>
inline cusparseStatus_t mat_vec_mul(cublasOperation_t transA,
                                    const sparse_mm_multGPU<Type> &A,
                                    const Type *x, Type *y,
                                    cudaStream_t stream = 0,
                                    bool mulChain = false) {
#ifdef PROFILING

	if (transA == CUBLAS_OP_N) {
		profile_info[2].valid = true;
		sprintf(profile_info[2].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 4 );
	} else {
		profile_info[3].valid = true;
		sprintf(profile_info[3].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 7 );
	}
	HANDLE_ERROR(cudaEventRecord(start));
#endif

	cusparseStatus_t ret_status;
	cusparseStatus_t *status = new cusparseStatus_t[A.numDevices];

	const Type **x_local = new const Type*[A.numDevices];
	Type **y_local = new Type*[A.numDevices];

	const Type alpha = 1;
	const Type beta = 0;

	int n, m, nnz;
	const sparse_mat_device<Type> *mA;

	cudaEvent_t inputStreamFinalize;
	HANDLE_ERROR(cudaEventCreate(&inputStreamFinalize));
	HANDLE_ERROR(cudaEventRecord(inputStreamFinalize, stream));

	cudaEvent_t *calcCompEvent = new cudaEvent_t[A.numDevices];

	int device = 0;

	Type *memAcc;

	if ((A.cscVertical == false) & (A.memAcc == NULL))
		// only happens if only one device is used for T-mul. Then, copy result 
		// directly to y, no acc neccesary
		memAcc = y;
	else
		memAcc = A.memAcc;

	for (int i = 0; i < A.numDevices; i++ ) {

		HANDLE_ERROR(cudaSetDevice(A.deviceArray[i]));
		HANDLE_ERROR(cudaEventCreate(&(calcCompEvent[i])));

		HANDLE_ERROR(cudaStreamWaitEvent(A.streams[i],
		                                 inputStreamFinalize, 0));

		// --- CUBLAS_OP_N ---
		if ((transA == CUBLAS_OP_N) & (i < A.numCSR)) {

			// Check if current device is main device: If so, no memory transfer 
			// necessary!
			if ( A.deviceArray[i] != A.mainDevice ) {

				// Set local pointers to pre-alloced memory
				x_local[i] = A.memA[i];
				y_local[i] = A.memB[i];

				// Broadcast vector x: copy complete vector from main device to all 
				// devices
				HANDLE_ERROR(cudaMemcpyAsync((void*)(x_local[i]), x, 
							A.A_csr[i]->dim_x * sizeof(Type), cudaMemcpyDeviceToDevice,
							A.streams[i]));

			} else {

				x_local[i] = x;
				y_local[i] = &y[A.csrRowIdx[i]];

			}

			// Do SpMV on sub matrices
			mA = A.A_csr[i];
			n = mA->dim_x;
			m = mA->dim_y;
			nnz = mA->nnz;

			HANDLE_ERROR(cudaMemsetAsync((void*)(y_local[i]), 0, m * sizeof(Type),
			                             A.streams[i]));

			HANDLE_ERROR(cusparseSetStream(A.cs_handle[i], A.streams[i]));
			status[i] = cusparse_csrmv(A.cs_handle[i],
					CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha,
					A.descrA[i], mA->val(), mA->ptr(), mA->ind(), x_local[i], &beta,
					y_local[i]);

			// Collect sub results: Copy vector y back to main device
			if ( A.deviceArray[i] != A.mainDevice ) {

				HANDLE_ERROR(cudaMemcpyAsync((void*)(&y[A.csrRowIdx[i]]), y_local[i],
							A.A_csr[i]->dim_y * sizeof(Type), cudaMemcpyDeviceToDevice,
							A.streams[i]));
			}

			// --- CUBLAS_OP_T ---
		} else if ((transA == CUBLAS_OP_T) & (i < A.numCSC)) {

			// Check if current device is main device: If so, no memory transfer 
			// necessary!
			if ( A.deviceArray[i] != A.mainDevice ) {

				// Set local pointers to pre-alloced memory
				if ( A.cscVertical == true ) {
					x_local[i] = A.memA[i];
					y_local[i] = A.memB[i];
				} else {
					x_local[i] = A.memB[i];
					y_local[i] = A.memA[i];
				}

				// Broadcast vector x: copy complete vector from main device to all 
				// devices
				if (A.cscVertical == true)
					HANDLE_ERROR(cudaMemcpyAsync((void*)(x_local[i]), x, 
								A.A_csc[i]->dim_y * sizeof(Type), cudaMemcpyDeviceToDevice,
								A.streams[i]));

				else if (mulChain == false)
					HANDLE_ERROR(cudaMemcpyAsync((void*)(x_local[i]), &x[A.cscColIdx[i]],
								A.A_csc[i]->dim_y * sizeof(Type), cudaMemcpyDeviceToDevice, 
								A.streams[i]));

			} else {

				if ( A.cscVertical == true ) {
					x_local[i] = x;
					y_local[i] = &y[A.cscColIdx[i]];
				} else {
					x_local[i] = &x[A.cscColIdx[i]];
					y_local[i] = y;
				}
			}

			// Do SpMV on sub matrices
			mA = A.A_csc[i];
			n = mA->dim_y;
			m = mA->dim_x;
			nnz = mA->nnz;

			HANDLE_ERROR(cudaMemsetAsync((void*)(y_local[i]), 0, m * sizeof(Type),
			                             A.streams[i]));

			HANDLE_ERROR(cusparseSetStream(A.cs_handle[i], A.streams[i]));
			status[i] = cusparse_csrmv(A.cs_handle[i],
					CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, A.descrA[i],
					mA->val(), mA->ptr(), mA->ind(), x_local[i], &beta, y_local[i]);

			// Collect sub/partial-results: Copy vector y back to main device
			if ( A.deviceArray[i] != A.mainDevice ) {

				if ( A.cscVertical == true ) {
					HANDLE_ERROR(cudaMemcpyAsync((void*)(&y[A.cscColIdx[i]]), y_local[i],
								A.A_csc[i]->dim_x * sizeof(Type), cudaMemcpyDeviceToDevice,
								A.streams[i]));
				} else {
					HANDLE_ERROR(cudaMemcpyAsync(
								(void*)(&memAcc[device * A.A_csc[i]->dim_x]), y_local[i],
								A.A_csc[i] ->dim_x * sizeof(Type), cudaMemcpyDeviceToDevice,
								A.streams[i]));
					device++;
				}

			}

		} else {
			status[i] = CUSPARSE_STATUS_SUCCESS; // if nothing is done, nothing can go wrong ;-)
		}

		HANDLE_ERROR(cudaEventRecord(calcCompEvent[i], A.streams[i]));
	}

	HANDLE_ERROR(cudaSetDevice(A.mainDevice));

	ret_status = CUSPARSE_STATUS_SUCCESS;

	for (int i = 0; i < A.numDevices; i++) {

		HANDLE_ERROR(cudaStreamWaitEvent(stream, calcCompEvent[i], 0));

		HANDLE_ERROR(cudaEventDestroy(calcCompEvent[i]));

		if ( status[i] != CUSPARSE_STATUS_SUCCESS ) {
			ret_status = status[i];
			printf("cuSparse %i error on device %i\n", status[i],
			       A.deviceArray[i]);
		}
	}

	if ((transA ==
	     CUBLAS_OP_T) & (A.cscVertical == false) & (A.numCSC > 1) ) {

		dim3 interAccThreads, interAccBlocks;
		interAccThreads.x = 32; //minimum number of threads per block

		dim3 volume(32, 32, 1);

		interAccBlocks.x = A.A_csc[0]->dim_x /
		                   (volume.y * volume.z * interAccThreads.x);
		interAccBlocks.y = volume.y / interAccThreads.y;
		interAccBlocks.z = volume.z / interAccThreads.z;

		if (A.mainUsedForSpMV == true)
			sum_vol_up_kernel_inter_wrapper(interAccBlocks, interAccThreads,
					interAccThreads.x * interAccThreads.y * interAccThreads.z *
					sizeof(Type), stream, A.numCSC - 1, true, A.A_csc[0]->dim_x /
					(volume.y * volume.z), volume.y, volume.z, y, memAcc);
		else
			sum_vol_up_kernel_inter_wrapper(interAccBlocks, interAccThreads,
					interAccThreads.x * interAccThreads.y * interAccThreads.z *
					sizeof(Type), stream, A.numCSC, false, A.A_csc[0]->dim_x /
					(volume.y * volume.z), volume.y, volume.z, y, memAcc);
	}

#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	if (transA == CUBLAS_OP_N) {
		profile_info[2].time += elapsedTime;
		profile_info[2].runs++;
	} else {
		profile_info[3].time += elapsedTime;
		profile_info[3].runs++;
	}
#endif

	return ret_status;
}
#endif

/******************************************************************************
* Multi-GPU processing for sparse matrices (multi-threaded initialization)
******************************************************************************/
#if (0)
/*
   void *thread_span_function (void *arg);
   int multithreaded_function (int deviceID, int mainDevice, sparse_mat_host *hA_csr, sparse_mat_host *hA_csc,           // input
                sparse_mat_device **A_csr, sparse_mat_device **A_csc, float **x_local, float **y_local,
                cudaStream_t *stream, cusparseHandle_t *cs_handle, cusparseMatDescr_t *descrA );

   //--------------------------------------------------------------------------
   // type definition for sparse matrices distributed over multiple GPUS
   //--------------------------------------------------------------------------
   class sparse_mm_multGPU_threaded {
   public:
        int mainDevice;
        int numDevices;
        int *deviceArray;
        int numCSC;
        int *cscColIdx;
        sparse_mat_device **A_csc;
        int *csrRowIdx;
        int numCSR;
        sparse_mat_device **A_csr;
        const float **x_local;
        float **y_local;
        cudaStream_t *streams;
        cusparseHandle_t *cs_handle;
        cusparseMatDescr_t *descrA;

        sparse_mm_multGPU_threaded(const sparse_mat_host &A, int numReqDevices, int *reqDeviceArray, int mainDevice);
        ~sparse_mm_multGPU_threaded();
        //void *thread_span_function (void *arg);

   };

   class pthread_handle_t {
   public:
        pthread_t pthread;
        int deviceID;
        int mainDevice;
        sparse_mat_host *hA_csr;
        sparse_mat_host *hA_csc;
        sparse_mat_device **A_csr;
        sparse_mat_device **A_csc;
        float **x_local;
        float **y_local;
        cudaStream_t *stream;
        cusparseHandle_t *cs_handle;
        cusparseMatDescr_t *descrA;

        pthread_handle_t ( int deviceID, int mainDevice, sparse_mat_host *hA_csr, sparse_mat_host *hA_csc,           // input
                        sparse_mat_device **A_csr, sparse_mat_device **A_csc, float **x_local, float **y_local,
                        cudaStream_t *stream, cusparseHandle_t *cs_handle, cusparseMatDescr_t *descrA ) :
                        deviceID(deviceID), mainDevice(mainDevice), hA_csr(hA_csr), hA_csc(hA_csc),
                        A_csr(A_csr), A_csc(A_csc), x_local(x_local), y_local(y_local),
                        stream(stream), cs_handle(cs_handle), descrA(descrA) {};


   };

   sparse_mm_multGPU_threaded::sparse_mm_multGPU_threaded(const sparse_mat_host &A, int numReqDevices, int *reqDeviceArray, int mainDevice) :
                mainDevice(mainDevice)  {

        double difftime;
        timeval start, stop;

        // Get names number and names of GPU devices ------------------------------------------------------------------
        gettimeofday(&start, NULL);

        int numAvailDevices = 0;
        cudaGetDeviceCount(&numAvailDevices);
        struct cudaDeviceProp *devices = (struct cudaDeviceProp*)malloc(numAvailDevices * sizeof(struct cudaDeviceProp) );

        int *canAccessPeer = new int[numAvailDevices];

        printf("Detected %i CUDA capable devices: ", numAvailDevices);

        for (int k=0; k < numAvailDevices; k++ ) {

                cudaGetDeviceProperties( &devices[k], k);
                printf("%s ", devices[k].name);

                cudaDeviceCanAccessPeer(&(canAccessPeer[k]), k, mainDevice);

        }

        printf("\n");

        if ( numAvailDevices < numReqDevices) {
                printf("WARNING: number of chosen GPUs exceeds maximum number! Reducing request to %i devices\n", numAvailDevices);
                numReqDevices = numAvailDevices;
        } else if ( numAvailDevices > numReqDevices) {
                printf("INFO: number of requested GPUs is smaller than number available\n", numReqDevices);
        }

        deviceArray = new int[numReqDevices];

        printf("Setting %s (device %i) as main device!\n", devices[mainDevice].name, mainDevice);

        numDevices = 0;
        for (int i=0; i < numReqDevices; i++) {

                if (( canAccessPeer[reqDeviceArray[i]] == 1 ) | ( reqDeviceArray[i] == mainDevice )) {

                        deviceArray[numDevices] = reqDeviceArray[i];
                        numDevices++;

                } else {

                        printf("Device %s can not be used as it can not access peer memory on main device!\n", devices[reqDeviceArray[i]].name);

                }

        }

        if ( numDevices < numReqDevices ) {

                printf("--> Actually processing on only %i devices, namely:", numDevices);

                for (int i=0; i< numDevices; i++)
                        printf(" %s", devices[deviceArray[i]].name);

                printf("\n");
        }

        size_t subMat_size;  // size in bytes
        subMat_size  = (2 * A.nnz * sizeof(float)) + (A.dim_y * sizeof(int)); // CSR matrix
        subMat_size += (2 * A.nnz * sizeof(float)) + (A.dim_x * sizeof(int)); // CSC matrix
        subMat_size = (size_t)ceil((double)subMat_size / (double)numDevices);
        if (subMat_size < 1024) {
                printf("Approx. mem-size for sub-matrices per GPU: %i Byte\n", subMat_size);
        } else if (subMat_size < 1024*1024) {
                printf("Approx. mem-size for sub-matrices per GPU: %i kByte\n", subMat_size / 1024);
        } else if (subMat_size >= 1024*1024) {
                printf("Approx. mem-size for sub-matrices per GPU: %i MByte\n", subMat_size / (1024 * 1024));
        }

        for (int k=0; k < numDevices; k++ ) {

                if ( devices[deviceArray[k]].totalGlobalMem < subMat_size ) {
                        printf("ERROR: not enough device memory for sub-matrices on at least one GPU!");
                        exit(-1);
                }

        }

        gettimeofday(&stop, NULL);
        difftime = (double)((stop.tv_sec * 1000000 + stop.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
        printf("time 1: %.3fms\n", difftime / 1000);

        // Generate content of sub-matrices -------------------------------------------------------------------------------
        gettimeofday(&start, NULL);

        csrRowIdx = new int[numDevices];
        cscColIdx = new int[numDevices];

        int nnzOpt = A.nnz / numDevices;
        if ( A.nnz * numDevices != nnzOpt ) // no integer devision
                nnzOpt    = ceil((double)A.nnz / (double)numDevices);

        int *nnzCSRAct = new int[numDevices]; //nnzOpt;
        int *nnzCSCAct = new int[numDevices]; //nnzOpt;

        int **buff_csr_ptr = new int*[numDevices];//[A.dim_x];
        int **buff_csc_ptr = new int*[numDevices];//[A.dim_y];

        const int **buff_csr_ind = new const int*[numDevices];
        const int **buff_csc_ind = new const int*[numDevices];

        const float **buff_csr_val = new const float*[numDevices];
        const float **buff_csc_val = new const float*[numDevices];

        int *csr_dim_y = new int[numDevices];
        int *csc_dim_x = new int[numDevices];

        int currCSRidx    = 0;
        int currCSRRowIdx = 0;
        int currCSRRowVal = A.csr_ptr()[currCSRRowIdx];

        int currCSCidx = 0;
        int currCSCColIdx = 0;
        int currCSCColVal = A.csc_ptr()[currCSCColIdx];

        numCSR = numDevices;
        numCSC = numDevices;

        sparse_mat_host **hA_csr = new sparse_mat_host*[numDevices];
        sparse_mat_host **hA_csc = new sparse_mat_host*[numDevices];

        for (int i=0; i < numDevices; i++) {

                // Generate and populate CSR sub-matrix if final block has not been reached
                if ( i < numCSR ) {

                        nnzCSRAct[i] = nnzOpt;
                        csrRowIdx[i] = currCSRRowIdx;

                        buff_csr_ind[i] = A.csr_ind() + currCSRidx;
                        buff_csr_val[i] = A.csr_val() + currCSRidx;
                        buff_csr_ptr[i] = new int[A.dim_y+1];

                        if ( currCSRidx + nnzCSRAct[i] >= A.nnz ) { // end of matrix reached with this block, make this block fitting and reduce numGPUs if whole device unused!

                                nnzCSRAct[i] = A.nnz - currCSRidx;
                                numCSR       = i + 1; //end of matrix is reached, do not use further GPUs (this should hopefully never happen)

                        } else {

                                while ( (A.csr_ind()[currCSRidx + nnzCSRAct[i] - 1] < A.csr_ind()[currCSRidx + nnzCSRAct[i]]) & (currCSRidx + nnzCSRAct[i] < A.nnz - 1) ) {
                                        nnzCSRAct[i]++;
                                }

                        }

                        int k = 0;
                        while ( (A.csr_ptr()[currCSRRowIdx + k] <= currCSRidx + nnzCSRAct[i]) & (currCSRRowIdx + k <= A.dim_y) ) {

                                buff_csr_ptr[i][k] = A.csr_ptr()[currCSRRowIdx + k] - currCSRRowVal;
                                k++;
                        }

                        k--; // last increment was too much!
                        csr_dim_y[i] = k;

                        // set values for next iteration
                        currCSRRowIdx += k;
                        currCSRRowVal = A.csr_ptr()[currCSRRowIdx];
                        currCSRidx   += nnzCSRAct[i];

                        hA_csr[i] = new sparse_mat_host(csr_dim_y[i], A.dim_x, nnzCSRAct[i], NULL, NULL, NULL, (float*)(buff_csr_val[i]), buff_csr_ptr[i], (int*)(buff_csr_ind[i]), sparse_mat_csr, false, 0);

                } else
                        hA_csc[i] = NULL;

                // Generate and populate CSC sub-matrix if final block has not been reached
                if ( i < numCSC ) {

                        nnzCSCAct[i] = nnzOpt;
                        cscColIdx[i] = currCSCColIdx;

                        // Generate and populate CSC sub-matrix
                        buff_csc_ind[i] = A.csc_ind() + currCSCidx;
                        buff_csc_val[i] = A.csc_val() + currCSCidx;
                        buff_csc_ptr[i] = new int[A.dim_x+1];

                        if ( currCSCidx + nnzCSCAct[i] >= A.nnz ) { // end of matrix reached with this block, make this block fitting and reduce numGPUs if whole device unused!

                                nnzCSCAct[i] = A.nnz - currCSCidx;
                                numCSC       = i + 1;

                        } else {

                                while ( (A.csc_ind()[currCSCidx + nnzCSCAct[i] - 1] < A.csc_ind()[currCSCidx + nnzCSCAct[i]]) & (currCSCidx + nnzCSCAct[i] < A.nnz - 1) ) {
                                        nnzCSCAct[i]++;
                                }

                        }

                        int k = 0;
                        while ( (A.csc_ptr()[currCSCColIdx+k] <= currCSCidx + nnzCSCAct[i]) & (currCSCColIdx + k <= A.dim_x) ) {

                                buff_csc_ptr[i][k] = A.csc_ptr()[currCSCColIdx + k] - currCSCColVal;
                                k++;
                        }

                        k--;
                        csc_dim_x[i] = k;

                        // set values for next iteration
                        currCSCColIdx += k;
                        currCSCColVal = A.csc_ptr()[currCSCColIdx];
                        currCSCidx   += nnzCSCAct[i];

                        hA_csc[i] = new sparse_mat_host(A.dim_y, csc_dim_x[i], nnzCSRAct[i], (float*)(buff_csc_val[i]), buff_csc_ptr[i], (int*)(buff_csc_ind[i]), NULL, NULL, NULL, sparse_mat_csc, false, 0);

                } else
                        hA_csc[i] = NULL;

        }

        numDevices = max(numCSR, numCSC); // if both sub-matrices do not span over all devices (due to division into full rows/cols)

        gettimeofday(&stop, NULL);
        difftime = (double)((stop.tv_sec * 1000000 + stop.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
        printf("time 2: %.3fms\n", difftime / 1000);

        // Generate content of sub-matrices -------------------------------------------------------------------------------
        gettimeofday(&start, NULL);

        cs_handle = new cusparseHandle_t[numDevices];
        descrA = new cusparseMatDescr_t[numDevices];
        streams = new cudaStream_t[numDevices];

        A_csc = new sparse_mat_device*[numDevices];
        A_csr = new sparse_mat_device*[numDevices];

        x_local = new const float*[numDevices];
        y_local = new float*[numDevices];

        pthread_handle_t **pthread_handle = new pthread_handle_t*[numDevices];

        // start multi-threaded processing
        for (int i=0; i < numDevices; i++) {

                pthread_handle[i] = new pthread_handle_t( deviceArray[i], mainDevice, hA_csr[i], hA_csc[i], &(A_csr[i]), &(A_csc[i]), (float**)(&(x_local[i])), &(y_local[i]),
                                &(streams[i]), &(cs_handle[i]), &(descrA[i]) );

                //create thread and start execution on spes (by calling ppu_thread_funktion)
                pthread_create(&(pthread_handle[i]->pthread), NULL, &thread_span_function, pthread_handle[i] );

        }

        //wait until all threads reach first barrier
        for (int i = 0; i < numDevices; i++ ) {

                //wait for threads to finish processing
                pthread_join(pthread_handle[i]->pthread, NULL);

        }

        gettimeofday(&stop, NULL);
        difftime = (double)((stop.tv_sec * 1000000 + stop.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
        printf("time 3: %.3fms(threaded)\n", difftime / 1000);

        HANDLE_ERROR(cudaSetDevice(mainDevice));

   }

   void *thread_span_function (void *arg) {

        pthread_handle_t *pthread_handle = (pthread_handle_t*)arg;

        multithreaded_function (pthread_handle->deviceID, pthread_handle->mainDevice, pthread_handle->hA_csr, pthread_handle->hA_csc,
                        pthread_handle->A_csr, pthread_handle->A_csc, pthread_handle->x_local, pthread_handle->y_local,
                        pthread_handle->stream, pthread_handle->cs_handle, pthread_handle->descrA );

        pthread_exit(NULL);
   }

   int multithreaded_function (int deviceID, int mainDevice, sparse_mat_host *hA_csr, sparse_mat_host *hA_csc,           // input
                sparse_mat_device **A_csr, sparse_mat_device **A_csc, float **x_local, float **y_local,
                cudaStream_t *stream, cusparseHandle_t *cs_handle, cusparseMatDescr_t *descrA ) {     // output

        int sizeX = 0;
        int sizeY = 0;

        HANDLE_ERROR(cudaSetDevice(deviceID));

        if ( hA_csr != NULL ) {

                (*A_csr) = new sparse_mat_device(*hA_csr, sparse_mat_csr);
                //cout << A_csr[i];

                sizeX = max(sizeX, (*A_csr)->dim_x);
                sizeY = max(sizeY, (*A_csr)->dim_y);

                //delete[] hA_csr;

        }

        if ( hA_csc != NULL ) {

                (*A_csc) = new sparse_mat_device(*hA_csc, sparse_mat_csc);
                //cout << A_csc[i];

                //delete[] hA_csc;

                sizeX = max(sizeX, (*A_csc)->dim_y);
                sizeY = max(sizeY, (*A_csc)->dim_x);

        }

        // Initialize necessary cuSparse handles
        HANDLE_ERROR(cusparseCreate(&(*cs_handle)));
        HANDLE_ERROR(cusparseCreateMatDescr(&(*descrA))); // general matrix, zero based indexing
        HANDLE_ERROR(cudaStreamCreate(&(*stream)));

        if ( deviceID != mainDevice ) {

                HANDLE_ERROR(cudaDeviceEnablePeerAccess(3, 0));

                HANDLE_ERROR(cudaMalloc((void**)&(*x_local), sizeX * sizeof(float)));
                HANDLE_ERROR(cudaMalloc((void**)&(*y_local), sizeY * sizeof(float)));

        }

        return 0;

   }

   sparse_mm_multGPU_threaded::~sparse_mm_multGPU_threaded() {

        for (int i=0; i < numDevices; i++) {

                if (( i < numCSR ) | ( i < numCSR )) {

                        HANDLE_ERROR(cusparseDestroyMatDescr(descrA[i]));
                        HANDLE_ERROR(cusparseDestroy(cs_handle[i]));
                        HANDLE_ERROR(cudaStreamDestroy(streams[i]));

                        cudaFree((void*)x_local[i]);
                        cudaFree(y_local[i]);

                }

                if ( i < numCSR )
                        delete(A_csc[i]);

                if ( i < numCSR )
                        delete(A_csr[i]);

        }

        delete(descrA);
        delete(cs_handle);

        delete(A_csr);
        delete(A_csc);

   }

   //-----------------------------------------------------------------------------------------------------------------------
   // mat_vec_mul for sparse matrices on distributed over multiple GPUS
   //-----------------------------------------------------------------------------------------------------------------------
   inline cusparseStatus_t mat_vec_mul(cublasOperation_t transA, const sparse_mm_multGPU_threaded &A, const float *x, float *y, cudaStream_t stream=0) {
   #ifdef PROFILING
        if (transA == CUBLAS_OP_N) {
                profile_info[2].valid = true;
                sprintf(profile_info[2].name, "%s (N)\t (%s,%i)\0", __FUNCTION__, __FILE__, __LINE__ - 4 );
        } else {
                profile_info[3].valid = true;
                sprintf(profile_info[3].name, "%s (T)\t (%s,%i)\0", __FUNCTION__, __FILE__, __LINE__ - 7 );
        }
        HANDLE_ERROR(cudaEventRecord(start));
   #endif

        cusparseStatus_t ret_status;
        cusparseStatus_t *status = new cusparseStatus_t[A.numDevices];

        const float **x_local = new const float*[A.numDevices];
        float **y_local = new float*[A.numDevices];

        const float alpha = 1;
        const float beta = 0;

        const double alpha0 = 1;
        const double beta0 = 0;

        int n, m, nnz;
        const sparse_mat_device *mA;

        cudaEvent_t *calcCompEvent = new cudaEvent_t[A.numDevices];

        for (int i=0; i < A.numDevices; i++ ) {

                HANDLE_ERROR(cudaSetDevice(A.deviceArray[i]));
                HANDLE_ERROR(cudaEventCreate(&(calcCompEvent[i])));

                if (( transA == CUBLAS_OP_N ) & ( i < A.numCSR )) {

                        // Check if current device is main device: If so, no memory transfer necessary!
                        if ( A.deviceArray[i] != A.mainDevice ) {

                                // Set local pointers to pre-alloced memory
                                x_local[i] = A.x_local[i];
                                y_local[i] = A.y_local[i];

                                // Broadcast vector x: copy complete vector from main device to all devices
                                HANDLE_ERROR(cudaMemcpyAsync((void*)(x_local[i]), x, A.A_csr[i]->dim_x * sizeof(float),
                                                cudaMemcpyDeviceToDevice, A.streams[i]));

                        } else {

                                x_local[i] = x;
                                y_local[i] = &y[A.csrRowIdx[i]];

                        }

                        // Do SpMV on sub-matrices
                        mA = A.A_csr[i];
                        n   = mA->dim_x;
                        m   = mA->dim_y;
                        nnz = mA->nnz;

                        HANDLE_ERROR(cusparseSetStream(A.cs_handle[i], A.streams[i]));
                        status[i] = cusparseScsrmv(A.cs_handle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, A.descrA[i],
                                        mA->val(), mA->ptr(), mA->ind(), x_local[i], &beta, y_local[i]);


                        // SpMV in double precision: only for test purposes
   //			double *Add, *xdd, *ydd;
   //			HANDLE_ERROR(cudaMalloc((void**)&Add, nnz * sizeof(double)));
   //			HANDLE_ERROR(cudaMalloc((void**)&xdd, n * sizeof(double)));
   //			HANDLE_ERROR(cudaMalloc((void**)&ydd, m * sizeof(double)));
   //
   //			convS2D_GPU_kernel_wrapper((int)ceil((double)nnz/1024.0), 1024, 0, A.streams[i], nnz,  Add, (float*)(mA->val()));
   //			convS2D_GPU_kernel_wrapper((int)ceil((double)n  /1024.0), 1024, 0, A.streams[i], n  ,  xdd, (float*)(x_local[i]));
   //
   //			HANDLE_ERROR(cusparseSetStream(A.cs_handle[i], A.streams[i]));
   //			status[i] = cusparseDcsrmv(A.cs_handle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha0, A.descrA[i],
   //					Add, mA->ptr(), mA->ind(), xdd, &beta0, ydd);
   //
   //			cudaFree(Add);
   //			cudaFree(xdd);
   //
   //			convD2S_GPU_kernel_wrapper((int)ceil((double)m  /1024.0), 1024, 0, A.streams[i], m, y_local[i], ydd);
   //			cudaFree(ydd);

                        // Collect sub-results: Copy vector y back to main device
                        if ( A.deviceArray[i] != A.mainDevice ) {

                                HANDLE_ERROR(cudaMemcpyAsync((void*)(&y[A.csrRowIdx[i]]), y_local[i], A.A_csr[i]->dim_y * sizeof(float),
                                                cudaMemcpyDeviceToDevice, A.streams[i]));

                        }

                } else if (( transA == CUBLAS_OP_T ) & ( i < A.numCSC )) {

                        // Check if current device is main device: If so, no memory transfer necessary!
                        if ( A.deviceArray[i] != A.mainDevice ) {

                                // Set local pointers to pre-alloced memory
                                x_local[i] = A.x_local[i];
                                y_local[i] = A.y_local[i];

                                // Broadcast vector x: copy complete vector from main device to all devices
                                HANDLE_ERROR(cudaMemcpyAsync((void*)(x_local[i]), x, A.A_csc[i]->dim_y * sizeof(float),
                                                cudaMemcpyDeviceToDevice, A.streams[i]));

                        } else {

                                x_local[i] = x;
                                y_local[i] = &y[A.cscColIdx[i]];

                        }

                        // Do SpMV on sub-matrices
                        mA = A.A_csc[i];
                        n   = mA->dim_y;
                        m   = mA->dim_x;
                        nnz = mA->nnz;

                        HANDLE_ERROR(cusparseSetStream(A.cs_handle[i], A.streams[i]));
                        status[i] = cusparseScsrmv(A.cs_handle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, A.descrA[i],
                                        mA->val(), mA->ptr(), mA->ind(), x_local[i], &beta, y_local[i]);


                        // SpMV in double precision: only for test purposes!
   //			double *Add, *xdd, *ydd;
   //			HANDLE_ERROR(cudaMalloc((void**)&Add, nnz * sizeof(double)));
   //			HANDLE_ERROR(cudaMalloc((void**)&xdd, n * sizeof(double)));
   //			HANDLE_ERROR(cudaMalloc((void**)&ydd, m * sizeof(double)));
   //
   //			convS2D_GPU_kernel_wrapper((int)ceil((double)nnz/1024.0), 1024, 0, A.streams[i], nnz,  Add, (float*)(mA->val()));
   //			convS2D_GPU_kernel_wrapper((int)ceil((double)n  /1024.0), 1024, 0, A.streams[i], n  ,  xdd, (float*)(x_local[i]));
   //
   //			HANDLE_ERROR(cusparseSetStream(A.cs_handle[i], A.streams[i]));
   //			status[i] = cusparseDcsrmv(A.cs_handle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha0, A.descrA[i],
   //					Add, mA->ptr(), mA->ind(), xdd, &beta0, ydd);
   //
   //			cudaFree(Add);
   //			cudaFree(xdd);
   //
   //			convD2S_GPU_kernel_wrapper((int)ceil((double)m  /1024.0), 1024, 0, A.streams[i], m, y_local[i], ydd);
   //			cudaFree(ydd);

                        // Collect sub-results: Copy vector y back to main device
                        if ( A.deviceArray[i] != A.mainDevice ) {

                                HANDLE_ERROR(cudaMemcpyAsync((void*)(&y[A.cscColIdx[i]]), y_local[i], A.A_csc[i]->dim_x * sizeof(float),
                                                cudaMemcpyDeviceToDevice, A.streams[i]));

                        }

                } else {

                        status[i] = CUSPARSE_STATUS_SUCCESS; // if nothing is done, nothing can go wrong ;-)
                }

                HANDLE_ERROR(cudaEventRecord(calcCompEvent[i], A.streams[i]));

        }

        ret_status = CUSPARSE_STATUS_SUCCESS;

        for (int i=0; i < A.numDevices; i++) {

                HANDLE_ERROR(cudaStreamWaitEvent(stream, calcCompEvent[i], 0));

                HANDLE_ERROR(cudaEventDestroy(calcCompEvent[i]));

                if ( status[i] != CUSPARSE_STATUS_SUCCESS ) {
                        ret_status = status[i];
                        printf("cuSparse %i error on device %i\n", status[i], A.deviceArray[i]);
                }

        }

        HANDLE_ERROR(cudaSetDevice(A.mainDevice));

   #ifdef PROFILING
        HANDLE_ERROR(cudaEventRecord(stop));
        float elapsedTime;
        HANDLE_ERROR(cudaEventSynchronize(stop));
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        if (transA == CUBLAS_OP_N) {
                profile_info[2].time += elapsedTime;
                profile_info[2].runs++;
        } else {
                profile_info[3].time += elapsedTime;
                profile_info[3].runs++;
        }
   #endif

        return ret_status;
   }

 */
#endif

/******************************************************************************
* Single-GPU processing for sparse matrices
******************************************************************************/
#if (1)
//#define _USE_ELL_ //makro for ELL / CSR/CSC switch

//-----------------------------------------------------------------------------
// type definition for sparse matrices on single GPU
//-----------------------------------------------------------------------------
template<class Type>
struct sparse_mm {
	sparse_mat_device<Type> *A_csc;
	sparse_mat_device<Type> *A_csr;
	cusparseHybMat_t A_ell_n;
	cusparseHybMat_t A_ell_t;
	cusparseHandle_t cs_handle;
	cusparseMatDescr_t descrA;

	sparse_mm(const sparse_mat_host<Type> &A, cusparseHandle_t handle,
	          cusparseMatDescr_t descriptor, cusparseHybMat_t ell_n,
	          cusparseHybMat_t ell_t) :
		A_ell_n(ell_n), A_ell_t(ell_t), cs_handle(handle), descrA(
		        descriptor) {

		size_t subMat_size;  // size in bytes

		// CSR matrix
		subMat_size = (2 * A.nnz * sizeof(Type)) + (A.dim_y * sizeof(int));
		// CSC matrix
		subMat_size += (2 * A.nnz * sizeof(Type)) + (A.dim_x * sizeof(int));        

		if (subMat_size < 1024) {
			printf("Mem-size of matrices on GPU: %i Byte\n",
			       subMat_size);
		} else if (subMat_size < 1024 * 1024) {
			printf("Mem-size of matrices on GPU: %i kByte\n",
			       subMat_size / 1024);
		} else if (subMat_size >= 1024 * 1024) {
			printf("Mem-size of matrices on GPU: %i MByte\n",
			       subMat_size / (1024 * 1024));
		}

		A_csc = new sparse_mat_device<Type>(A, sparse_mat_csc);
		A_csr = new sparse_mat_device<Type>(A, sparse_mat_csr);

#ifdef _USE_ELL_

#ifdef PROFILING
		profile_info[0].valid = true;
		sprintf(profile_info[0].name, "%s\t (%s,%i)\0",
		        "cusparseScsr2hyb(N)", __FILE__, __LINE__);
		HANDLE_ERROR(cudaEventRecord(start));
#endif

		// use csr matrix for generation of ell_n matrix
		HANDLE_ERROR(cusparseScsr2hyb(cs_handle, A.dim_y, A.dim_x,
		                              descrA, A_csr->val(),
		                              A_csr->ptr(), A_csr->ind(),
		                              A_ell_n, 0, CUSPARSE_HYB_PARTITION_AUTO));

#ifdef PROFILING
		HANDLE_ERROR(cudaEventRecord(stop));
		float elapsedTime;
		HANDLE_ERROR(cudaEventSynchronize(stop));
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
		profile_info[0].time += elapsedTime;
		profile_info[0].runs++;
#endif
		memcheck(0); // 2480 MB --> A_ell_n: 158 MB

#ifdef PROFILING
		profile_info[1].valid = true;
		sprintf(profile_info[1].name, "%s\t (%s,%i)\0",
		        "cusparseScsr2hyb(T)", __FILE__, __LINE__);
		HANDLE_ERROR(cudaEventRecord(start));
#endif

		// use csr matrix for generation of ell_t matrix
		HANDLE_ERROR(cusparseScsr2hyb(cs_handle, A.dim_x, A.dim_y,
		                              descrA, A_csr->val(),
		                              A_csc->ptr(), A_csc->ind(),
		                              A_ell_t, 0,
		                              CUSPARSE_HYB_PARTITION_AUTO));

#ifdef PROFILING
		HANDLE_ERROR(cudaEventRecord(stop));
		HANDLE_ERROR(cudaEventSynchronize(stop));
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
		profile_info[1].time += elapsedTime;
		profile_info[1].runs++;
#endif

		memcheck(0); // 2291 MB --> A_ell_t: 189 MB

		delete(A_csc);
		delete(A_csr);

#endif
	};
};

//-----------------------------------------------------------------------------
// mat_vec_mul for sparse matrices
//-----------------------------------------------------------------------------
template<class Type>
inline cusparseStatus_t mat_vec_mul(cublasOperation_t transA,
		const sparse_mm<Type> &A, const Type *x, Type *y, cudaStream_t stream = 0,
		bool mulChain = false) {

#ifdef PROFILING
	if (transA == CUBLAS_OP_N) {
		profile_info[2].valid = true;
		sprintf(profile_info[2].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 4 );
	} else {
		profile_info[3].valid = true;
		sprintf(profile_info[3].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 7 );
	}
	HANDLE_ERROR(cudaEventRecord(start));
#endif

	cusparseStatus_t status;

#ifndef _USE_ELL_
	int n, m;
	const sparse_mat_device<Type> *mA;

	if (transA == CUBLAS_OP_N) {
		n = A.A_csr->dim_x;
		m = A.A_csr->dim_y;
		mA = A.A_csr;
	} else {
		n = A.A_csr->dim_y;
		m = A.A_csr->dim_x;
		mA = A.A_csc; // implicitly transposed!
	}

	int nnz = mA->nnz;

	const Type alpha = 1;
	const Type beta = 0;

	HANDLE_ERROR(cusparseSetStream(A.cs_handle, stream));
	status = cusparse_csrmv(A.cs_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
	                        m, n, nnz, &alpha, A.descrA, mA->val(),
	                        mA->ptr(), mA->ind(), x, &beta, y);

#else
	HANDLE_ERROR(cusparseSetStream(A.cs_handle, stream));

	const Type alpha = 1;
	const Type beta = 0;

	if (transA == CUBLAS_OP_N) {

		status = cusparse_hybmv(A.cs_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
				&alpha, A.descrA, A.A_ell_n, x, &beta, y);
	} else {

		status = cusparse_hybmv(A.cs_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
				&alpha, A.descrA, A.A_ell_t, x, &beta, y);
	}
#endif

#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	if (transA == CUBLAS_OP_N) {
		profile_info[2].time += elapsedTime;
		profile_info[2].runs++;
	} else {
		profile_info[3].time += elapsedTime;
		profile_info[3].runs++;
	}
#endif

	return status;
}
#endif

#endif /* MAT_VEC_MUL_SPARSE_H_ */
