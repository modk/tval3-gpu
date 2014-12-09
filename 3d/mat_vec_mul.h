#ifndef MAT_VEC_MUL_H_
#define MAT_VEC_MUL_H_

#include "profile_gpu.h"
#include "handle_error.h"
#include "time.h"
#include "sys/time.h"
#include <stdio.h>

using namespace std;

void memcheck (int i, bool newLine = true) {

	int currDevice;
	cudaGetDevice(&currDevice);

	size_t totalmem, freemem;
	cudaSetDevice(i);
	cudaMemGetInfo( &freemem, &totalmem );
	printf("total: %i MB, free: %i MB on device %i", (size_t)totalmem /
	       (1024 * 1024), (size_t)freemem / (1024 * 1024), i );

	if ( newLine == true)
		printf("\n");

	cudaSetDevice(currDevice);

}

void memcheck () {

	size_t totalmem, freemem;
	int device;
	cudaGetDevice(&device);
	cudaMemGetInfo( &freemem, &totalmem );
	printf("total: %i MB, free: %i MB on device %i\n", (size_t)totalmem /
	       (1024 * 1024), (size_t)freemem / (1024 * 1024), device );

}

struct matDim {
	int min;
	int max;

	matDim() : min(0), max(0) {
	};
};

template<typename Type>
struct transType {
	bool initDone;
	matDim *subDims;
	Type ** memA;
	Type ** memB;

	transType() : initDone (false), subDims(NULL), memA(NULL), memB(NULL) {
	};
};

#include "tval3_gpu_3d_kernels.cuh"

#include "mat_vec_mul_dense.h"
#include "mat_vec_mul_sparse.h"
#include "mat_vec_mul_dynamic.h"


/****************************************************************************************************************************************************
* Multi-GPU processing for dynamically calculated matrices for N and sparse matrices for T multiply
   /***************************************************************************************************************************************************/
#if (1)
//-----------------------------------------------------------------------------------------------------------------------
// type definition for combined geom_mm and sparse_mm_multGPU
//-----------------------------------------------------------------------------------------------------------------------
template <class Type>
struct sparse_geom_mm {
	// common
	int mainDevice;
	int numDevices;
	bool mainUsedForSpMV;
	int *deviceArray;
	cudaStream_t *streams;
	Type **memA;
	Type **memB;
	Type  *memAcc;
	// geometry specific
	geom_mm_multGPU<Type> *geo;
	geometry_device **dA_geo;
	textureParams<Type> **params;
	// sparse matrix specific
	sparse_mm_multGPU<Type> *mat;
	int numCSC;
	int *cscColIdx;
	sparse_mat_device<Type> **dA_mat;
	cusparseHandle_t *cs_handle;
	cusparseMatDescr_t *descrA;

	sparse_geom_mm(const geometry_host &hA_geo,
	               const sparse_mat_host<Type> &hA_mat, int numReqDevices,
	               int *reqDeviceArray, int mainDevice);
	~sparse_geom_mm();

};

template <class Type>
sparse_geom_mm<Type>::sparse_geom_mm(const geometry_host &hA_geo,
                                     const sparse_mat_host<Type> &hA_mat,
                                     int numReqDevices, int *reqDeviceArray,
                                     int mainDevice) :
	mainDevice(mainDevice) {

	double difftime;
	timeval start, stop, start0, stop0;

	printf(
	        "---------------------------------- geo & sparse init ----------------------------------\n");

	mainUsedForSpMV = false;

	// Get names number and names of GPU devices --------------------------------
	gettimeofday(&start, NULL);
	start0 = start;

	int numAvailDevices = 0;
	cudaGetDeviceCount(&numAvailDevices);
	cudaDeviceProp *devices = new cudaDeviceProp[numAvailDevices];

	int *canAccessPeer = new int[numAvailDevices];

	printf("Detected %i CUDA capable devices: ", numAvailDevices);

	for (int k = 0; k < numAvailDevices; k++ ) {

		cudaGetDeviceProperties( &devices[k], k);
		printf("%s ", devices[k].name);

		cudaDeviceCanAccessPeer(&(canAccessPeer[k]), k, mainDevice);

	}

	printf("\n");

	if ( numAvailDevices < numReqDevices) {
		printf(
		        "WARNING: number of chosen GPUs exceeds maximum number! Reducing request to %i devices\n",
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
			printf("Device %s can not be used as it can not access peer memory on "\ 
					"main device!\n", devices[reqDeviceArray[i]].name);
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

	gettimeofday(&stop, NULL);
	difftime =
	        (double)((stop.tv_sec * 1000000 +
	                  stop.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
	//printf("time 1: %.3fms\n", difftime / 1000);

	// Determine optimal number of signals per submatrix ------------------------
	gettimeofday(&start, NULL);

	int avgNnzPerRow = ceil((double)hA_mat.nnz / (double)hA_mat.dim_y);
	int avgMemPerRow = avgNnzPerRow * (sizeof(int) + sizeof(Type));

	int pathsMainOpt = 0, pathsOpt = 0;

	double mainOverFactor = 1; //1.25; // must be < max(2, numDevices) TODO: find dependence on numDevices!

	if ((mainUsedForSpMV == true) & (numDevices > 1)) {

		// Determine free memory on main device -----------------------------------
		int currDevice;
		HANDLE_ERROR(cudaGetDevice(&currDevice));
		size_t totalMem, freeMem;
		HANDLE_ERROR(cudaSetDevice(mainDevice));
		HANDLE_ERROR(cudaMemGetInfo( &freeMem, &totalMem ));
		HANDLE_ERROR(cudaSetDevice(currDevice));

		// subtract memory needed for further matrices on main GPU alloced in main 
		// routine (numbers are constant, only dependent on resolution/signals)
		freeMem -= (28 + 4 + 3) * hA_mat.dim_x * sizeof(Type) +
		        (7 + 1 + 1) * hA_mat.dim_y * sizeof(Type);

		// subtract additional memory needed for accumulation of partial volumes in T-multiply
		freeMem -= hA_mat.dim_x * (numDevices - 1) * sizeof(Type);

		// subtract additional memory for geometries structs on device
		freeMem -=
		        (3 * (hA_geo.num_emitters + hA_geo.num_receivers) * hA_geo.numMPs *
		         sizeof(int)) + (hA_mat.dim_y * sizeof(uint2));

		// subtract additional memory texture fetching 3Dmem + array
		if ( hA_geo.rv_y * sizeof(Type) <= 512 ) // memory depends on pitch
			freeMem -= hA_geo.rv_z * hA_geo.rv_x * 512;
		else if ( hA_geo.rv_y * sizeof(Type) <= 1024 )
			freeMem -= hA_geo.rv_z * hA_geo.rv_x * 1024;
		else //if ( hA_geo.rv_y * sizeof(Type) <= 2048 )
			freeMem -= hA_geo.rv_z * hA_geo.rv_x * 2048;

		freeMem -= hA_mat.dim_x * sizeof(Type) * 9; //experimentally determined value for 3d-array

		// determine maximum number of rows on main device
		// ASSUMPTION: equal distribution of nnzs per row (true for #rows >> #receivers)

		freeMem -= hA_mat.dim_x * sizeof(int); //ptr-array independent of number of rows/nnzs

		pathsMainOpt =
		        floor((double)freeMem /
		              (double)avgMemPerRow) - hA_geo.numPaths /
		        hA_geo.num_emitters *
		        emitterPerTas;
		pathsOpt = ceil(
		        (double)(hA_mat.dim_y - pathsMainOpt) / (double)(numDevices - 1));

		// check if nnzMainOpt is by far too large (due to small matrix size or large free GPU memory)
		double mainOverFactor = 1; //1.25; // must be < numDevices

		if (pathsMainOpt > hA_mat.dim_y) { //complete matrix would fit onto device

			pathsMainOpt = floor((double)(hA_mat.dim_y) /
			              (double)(numDevices + mainOverFactor - 1) * mainOverFactor);
			pathsOpt = ceil (
			        (double)(hA_mat.dim_y - pathsMainOpt) / (double)(numDevices - 1));

		} else if ( (double)pathsMainOpt / mainOverFactor >
		            (double)pathsOpt ) {  // if mainOverFactor more NNZs

			pathsMainOpt =
			        floor((double)(hA_mat.dim_y) /
			              (double)(numDevices + mainOverFactor -
			                       1) * mainOverFactor);
			pathsOpt = ceil (
			        (double)(hA_mat.dim_y -
			                 pathsMainOpt) /
			        (double)(numDevices - 1));

		} else if (pathsMainOpt < pathsOpt) {

			printf("Reduced paths on main device due to lack of memory: " \ 
					"pathsMainOpt=%i, pathsOpt=%i, pathsMainOpt/pathsOpt=%.3f\n",
			        pathsMainOpt, pathsOpt, (double)pathsMainOpt /
			        (double)pathsOpt);
		}

	} else {

		pathsMainOpt = ceil((float)(hA_mat.dim_y) / (double)numDevices);
		pathsOpt = pathsMainOpt;

	}

	gettimeofday(&stop, NULL);
	difftime =
	        (double)((stop.tv_sec * 1000000 +
	                  stop.tv_usec) -
	                 (start.tv_sec * 1000000 + start.tv_usec));
	//printf("time 2: %.3fms\n", difftime / 1000);

	// Generate data elements for sub-geometries  -------------------------------
	gettimeofday(&start, NULL);

	int numTotalSignals = hA_geo.num_emitters * hA_geo.num_receivers *
	                      hA_geo.numMPs;
	int numSignalsPerMP = hA_geo.num_emitters * hA_geo.num_receivers;

	int numTotalPath = hA_geo.numPaths * hA_geo.numMPs;
	int numPathPerMP = hA_geo.numPaths;

	int currPath = 0, totPath = 0, currSignal = 0;
	int path_tmp = 0, path_val = 0, last_path_val = 0;
	int emitter_tmp = 0, idx_tmp = 0, tas_tmp = 0;

	unsigned int  *num_signals = new unsigned int [numDevices];
	unsigned int  *num_path = new unsigned int [numDevices];

	uint2 *global_path_array = new uint2[numTotalPath];
	uint2 **use_path = new uint2*[numDevices];

	int numActDevices = 0, pathsPerGPUopt = 0;

	if (deviceArray[numActDevices] == mainDevice)
		pathsPerGPUopt = pathsMainOpt;
	else
		pathsPerGPUopt = pathsOpt;

	cscColIdx = new int[numDevices];

	int i = 0;
	while (i < numTotalSignals ) {

		if ((currPath >=
		     pathsPerGPUopt) & (i % (emitterPerTas * hA_geo.num_receivers) == 0)) {

			cscColIdx[numActDevices] = path_tmp;

			use_path[numActDevices] =
			        &(global_path_array[path_tmp]);
			path_tmp = totPath;

			num_signals[numActDevices] = i - idx_tmp;
			idx_tmp = i;

			num_path[numActDevices] = currPath;
			currPath = 0;

			numActDevices++;

			if ( deviceArray[numActDevices] == mainDevice )
				pathsPerGPUopt = pathsMainOpt;
			else
				pathsPerGPUopt = pathsOpt;

			last_path_val = path_val;

		} else { // ( numCurrSignals < signalsPerGPUopt) | // walk through use_path vector and icremenent numCurrSignals whenever use_path != 0
			// ( i % (emitterPerTas * A_host.num_receivers) != 0) {  // if numCurrSignals has reached signalsPerGPUopt, then go further until TAS boarder is reached
			// (increment numCurrSignals if meanwhile further valid paths are detected)

			if ( hA_geo.use_path[i % numSignalsPerMP] != 0) {

				global_path_array[totPath].x = i /
				                               hA_geo.
				                               num_receivers;
				global_path_array[totPath].y =
				        (i /
				         (numSignalsPerMP)) *
				        hA_geo.num_receivers +
				        (i % hA_geo.num_receivers);

				currPath++;
				totPath++;
			}

			i++;
		}

	}

	if ( currPath > 0) {

		cscColIdx[numActDevices] = path_tmp;

		use_path[numActDevices] = &(global_path_array[path_tmp]);

		num_signals[numActDevices] = i - idx_tmp;
		num_path[numActDevices] = currPath;

		numActDevices++;

	}

	numDevices = numActDevices;

	gettimeofday(&stop, NULL);
	difftime = (double)((stop.tv_sec * 1000000 + stop.tv_usec) - 
			(start.tv_sec * 1000000 + start.tv_usec));
	//printf("time 3: %.3fms\n", difftime / 1000);

	// Generate data elements for sub-matrices  ---------------------------------
	gettimeofday(&start, NULL);

	numCSC = numDevices;

	int *nnzCSCAct = new int[numDevices];;
	int **buff_csc_ptr = new int*[numDevices];
	int **buff_csc_ind = new int *[numDevices];
	Type **buff_csc_val = new Type*[numDevices];

	int *csc_dim_x = new int[numDevices];
	int *csc_dim_y = new int[numDevices];;

	// Generate and populate CSC sub-matrix for innovative horizontal cut 
	// (along volume axis, X): approx. equal number of NNZs and rows per 
	// submatrix, but needs accumulation for T-multiply
	if (numCSC > 1) {

		int *currIdx = new int[numCSC];
		matDim *dimY = new matDim[numCSC];

		dimY[0].min = 0;

		for (int i = 0; i < numCSC; i++) {

			csc_dim_x[i] = hA_mat.dim_x;
			csc_dim_y[i] = num_path[i];
			buff_csc_ptr[i] = new int [hA_mat.dim_x + 1];

			dimY[i].max = dimY[i].min + csc_dim_y[i];

			if (i < numCSC - 1) dimY[i + 1].min = dimY[i].max;

		}

		int currRow = 0, currDevice = 0, currCol = 0;

		bool done = false;
		int k = 0;
		while ( done == false ) {

			for (int i = 0; i < numCSC; i++) {

				nnzCSCAct[i] = (1 + (k + 1) * 0.2) * avgNnzPerRow * csc_dim_y[i];

				if (k != 0) {

					delete(buff_csc_ind[i]);
					delete(buff_csc_val[i]);
				}

				buff_csc_ind[i] = new int [nnzCSCAct[i]];
				buff_csc_val[i] = new Type[nnzCSCAct[i]];

				currIdx[i] = 0;
				buff_csc_ptr[i][0] = 0;

			}

			currRow = 0; currDevice = 0; currCol = 0;

			int i = 0;
			while ((i < hA_mat.nnz) & (currIdx[currDevice] < nnzCSCAct[currDevice]) ) {

				currRow = hA_mat.csc_ind()[i];

				if ( (currRow <
				      dimY[currDevice].max) &
						// simply add element to matrix and go on
				     (i < hA_mat.csc_ptr()[currCol + 1]) ) { 

					buff_csc_ind[currDevice][currIdx[ currDevice ]] = 
						currRow - dimY[currDevice].min;

					buff_csc_val[currDevice][currIdx[currDevice]] = hA_mat.csc_val()[i];
					currIdx[currDevice]++;

					i++;

				} else { // slice of current device passed, try next one or go to next column

					// finalize column for this device
					buff_csc_ptr[currDevice][currCol +
					                         1] =
					        currIdx[currDevice];

					if (currDevice < numCSC - 1) {
						currDevice++;
					} else {
						currDevice = 0;
						currCol++;
					}
				}
			}

			if (i == hA_mat.nnz) {
				done = true;
				k++;
				for (int v = 0; v < numCSC; v++)
					nnzCSCAct[v] = currIdx[v];
			}
		}

		// finalize very remaining columns
		while (currCol < csc_dim_x[currDevice] ) {

			buff_csc_ptr[currDevice][currCol + 1] = currIdx[currDevice];

			if (currDevice < numCSC - 1) {
				currDevice++;
			} else {
				currDevice = 0;
				currCol++;
			}
		}

		delete[] currIdx;
		delete[] dimY;

	} else {

		nnzCSCAct[0] = hA_mat.nnz;
		csc_dim_x[0] = hA_mat.dim_x;
		csc_dim_y[0] = hA_mat.dim_y;

		buff_csc_ptr[0] = (int*)hA_mat.csc_ptr();
		buff_csc_ind[0] = (int*)hA_mat.csc_ind();
		buff_csc_val[0] = (Type*)hA_mat.csc_val();

		cscColIdx[0] = 0;

	}

	gettimeofday(&stop, NULL);
	difftime =
	        (double)((stop.tv_sec * 1000000 +
	                  stop.tv_usec) -
	                 (start.tv_sec * 1000000 + start.tv_usec));
	//printf("time 4: %.3fms\n", difftime / 1000);

	// Generate handles and device matrices, copy data to devices ---------------
	gettimeofday(&start, NULL);

	cs_handle = new cusparseHandle_t[numDevices];
	descrA = new cusparseMatDescr_t[numDevices];
	streams = new cudaStream_t[numDevices];

	dA_geo = new geometry_device*[numDevices];
	dA_mat = new sparse_mat_device<Type>*[numDevices];

	memA = new Type*[numDevices];
	memB = new Type*[numDevices];
	params = new textureParams<Type>*[numDevices];

	printf("numDevices=%i, nnz=%i, dim_x=%i, dim_y=%i, numTotalSignals=%i, " \ 
			"numTotalPaths=%i\n", numDevices, hA_mat.nnz, hA_mat.dim_x, hA_mat.dim_y,
	        numTotalSignals, hA_geo.numPaths * hA_geo.numMPs);

	for (int i = 0; i < numDevices; i++) {

		int path = 0;

		if (deviceArray[i] != mainDevice )
			path = pathsOpt;
		else
			path = pathsMainOpt;

		HANDLE_ERROR(cudaSetDevice(deviceArray[i]));

		if (deviceArray[i] != mainDevice )
			HANDLE_ERROR(cudaDeviceReset());

		dA_mat[i] =
		        new sparse_mat_device<Type>(csc_dim_y[i], csc_dim_x[i],
		                                    nnzCSCAct[i],
		                                    buff_csc_val[i],
		                                    buff_csc_ptr[i],
		                                    buff_csc_ind[i],
		                                    sparse_mat_csc);

		delete[] buff_csc_ptr[i];
		delete[] buff_csc_ind[i];
		delete[] buff_csc_val[i];

		printf("GPU[%i]: nnz=%i (%2.1f\%), dim_x=%i (%2.1f\%), " \ 
				"dim_y=%i (%+i, %2.1f\%), ",
		        deviceArray[i], dA_mat[i]->nnz,
		        (dA_mat[i]->nnz / (float)hA_mat.nnz) * 100,
		        dA_mat[i]->dim_x,
		        (dA_mat[i]->dim_x / (float)hA_mat.dim_x) * 100,
		        dA_mat[i]->dim_y, dA_mat[i]->dim_y - path,
		        (dA_mat[i]->dim_y / (float)hA_mat.dim_y) * 100);

		dA_geo[i] = new geometry_device(hA_geo, (unsigned int*)(use_path[i]),
		                            num_path[i], num_signals[i]);
		params[i] = new textureParams<Type>(hA_geo.rv_x, hA_geo.rv_y,
		                                    hA_geo.rv_z, 0);

		printf("numSignals=%i (%2.1f \%), numPaths=%i (%+i, %2.1f \%)\n",
		        num_signals[i], num_signals[i] / (float)numTotalSignals * 100,
		        num_path[i], num_path[i] - path, num_path[i] / (float)numTotalPath *
		        100);

		// Initialize necessary cuSparse handles
		HANDLE_ERROR(cusparseCreate(&(cs_handle[i])));
		// general matrix, zero based indexing
		HANDLE_ERROR(cusparseCreateMatDescr(&(descrA[i]))); 		
		HANDLE_ERROR(cudaStreamCreate(&(streams[i])));

		if ( deviceArray[i] != mainDevice ) {

			HANDLE_ERROR(cudaDeviceEnablePeerAccess(mainDevice, 0));

			HANDLE_ERROR(cudaMalloc((void**)&(memA[i]),
			                        (int)hA_mat.dim_x * sizeof(Type) ));
			HANDLE_ERROR(cudaMalloc((void**)&(memB[i]), num_path[i] * sizeof(Type) ));

		}
	}

	HANDLE_ERROR(cudaSetDevice(mainDevice));

	if (numDevices > 1) {

		if ( mainUsedForSpMV == true)
			HANDLE_ERROR(cudaMalloc((void**)(&memAcc),
			                        (int)hA_mat.dim_x * (numDevices -
			                         1) * sizeof(Type)));
		else
			HANDLE_ERROR(cudaMalloc((void**)(&memAcc),
			                        (int)hA_mat.dim_x * numDevices * sizeof(Type)));

	} else { 
		// (numCSC == 1) --> only one device, no buffer needed. memTmp will be set to result in mul-routine
		memAcc = NULL;
	}

	for (int i = 0; i < numDevices; i++) {

		size_t subMat_size =
		        (dA_mat[i]->nnz * sizeof(Type)) + (dA_mat[i]->nnz * sizeof(int)) + 
						(dA_mat[i]->dim_x * sizeof(int)); // CSC matrix

		if (subMat_size < 1024) {
			printf("Mem-size for sub-matrices on GPU %i: %i Byte",
			       deviceArray[i], subMat_size );
		} else if (subMat_size < 1024 * 1024) {
			printf("Mem-size for sub-matrices on GPU %i: %i kByte",
			       deviceArray[i], subMat_size / 1024);
		} else if (subMat_size >= 1024 * 1024) {
			printf("Mem-size for sub-matrices on GPU %i: %i MByte",
			       deviceArray[i], subMat_size / (1024 * 1024));
		}

		printf(" ("); memcheck(deviceArray[i], false); printf(")\n");
	}

	geo = new geom_mm_multGPU<Type>(mainDevice, numDevices, mainUsedForSpMV,
	                                deviceArray, streams, 0,
	                                dA_geo, params, memA, memB, NULL, NULL);

	mat = new sparse_mm_multGPU<Type>(mainDevice, numDevices, false,
	                                  mainUsedForSpMV, deviceArray, numCSC,
	                                  cscColIdx,
	                                  dA_mat, NULL, 0, NULL, memA, memB,
	                                  memAcc, streams, cs_handle, descrA);

	gettimeofday(&stop, NULL);
	difftime = (double)((stop.tv_sec * 1000000 + stop.tv_usec) - 
			(start.tv_sec * 1000000 + start.tv_usec));
	//printf("time 5: %.3fms\n", difftime / 1000);


	// Finalize: clear intermediate arrays and set context (back to) main device 
	delete[] devices;               devices = NULL;
	delete[] canAccessPeer;         canAccessPeer = NULL;
	delete[] num_signals;           num_signals = NULL;
	delete[] num_path;              num_path = NULL;
	delete[] global_path_array;     global_path_array = NULL;
	delete[] use_path;              use_path = NULL;

	delete[] nnzCSCAct;             nnzCSCAct = NULL;
	delete[] buff_csc_ptr;          buff_csc_ptr = NULL;
	delete[] buff_csc_ind;          buff_csc_ind = NULL;
	delete[] buff_csc_val;          buff_csc_val = NULL;
	delete[] csc_dim_x;             csc_dim_x = NULL;
	delete[] csc_dim_y;             csc_dim_y = NULL;

	gettimeofday(&stop0, NULL);
	difftime = (double)((stop0.tv_sec * 1000000 + stop0.tv_usec) -
	                 (start0.tv_sec * 1000000 + start0.tv_usec));
	printf("Time for GPU init: %.3fms\n", difftime / 1000);
}

template <class Type>
sparse_geom_mm<Type>::~sparse_geom_mm() {

	for (int i = 0; i < numDevices; i++) {

		HANDLE_ERROR(cudaSetDevice(deviceArray[i]));

		HANDLE_ERROR(cudaStreamDestroy(streams[i]));
		HANDLE_ERROR(cusparseDestroyMatDescr(descrA[i]));
		HANDLE_ERROR(cusparseDestroy(cs_handle[i]));

		if (memA[i] != NULL) cudaFree(memA[i]);

		if (memB[i] != NULL) cudaFree(memA[i]);

		delete dA_mat[i];
		delete dA_geo[i];
		delete params[i];
	}

	HANDLE_ERROR(cudaSetDevice(mainDevice));

	if (memAcc != NULL) cudaFree(memAcc);

	delete[] memA;
	delete[] memB;

	delete[] dA_geo;
	delete[] params;
	delete[] dA_mat;

	delete[] cscColIdx;
	delete[] deviceArray;

	delete mat;
	delete geo;

}


//-----------------------------------------------------------------------------------------------------------------------
// matrix-vector multiplication for dyn-calc-N / sparse-T multiply
//-----------------------------------------------------------------------------------------------------------------------
template<class Type>
inline cublasStatus_t mat_vec_mul(cublasOperation_t trans,
                                  const sparse_geom_mm<Type> &A, const Type *x,
                                  Type *y, cudaStream_t stream = 0,
                                  bool mulChain = false) {

	cublasStatus_t status;

	if (trans == CUBLAS_OP_N) {

		// use dynamic calculation for N-multiply:
		status = mat_vec_mul(CUBLAS_OP_N, *(A.geo), x, y, stream,
		                     mulChain);

	} else {

		// use cusparse for T-multiply
		status = (cublasStatus_t)mat_vec_mul(CUBLAS_OP_T, *(A.mat), x,
		                                     y, stream, mulChain);

	}

	return status;

}
#endif

#endif /* MAT_VEC_MUL_H_ */

