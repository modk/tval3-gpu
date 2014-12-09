#ifndef MAT_VEC_MUL_DYNAMIC_H_
#define MAT_VEC_MUL_DYNAMIC_H_

/*******************************************************************************
* Single-GPU processing for dynamically calculated matrices with hierarchical 
* T-multiply
*******************************************************************************/

#if (0)
//-----------------------------------------------------------------------------
// type definition for geometry struct and hierarchical T-multiply
//-----------------------------------------------------------------------------
template<class Type>
struct geom_mm {
	geometry_device A;
	Type *y_local;
	int sub_vol;
	textureParams *params;
	geom_mm(const geometry_host &A_host,
	        int sub_vol = 1) : A(A_host), sub_vol(sub_vol) {

		y_local = NULL;

		params = new textureParams(A_host.rv_x, A_host.rv_y,
		                           A_host.rv_z, sub_vol);

		if ( sub_vol > 0 )
			HANDLE_ERROR(cudaMalloc((void**)&y_local, sub_vol *
			                        A_host.rv_x * A_host.rv_y *
			                        A_host.rv_z * sizeof(Type)));
	}

	~geom_mm() {
		if (y_local != NULL) cudaFree(y_local);
	}
};

//-----------------------------------------------------------------------------
// matrix-vector multiplication with dynamic calculation of the measurement 
// matrix (kernel moved to separate file)
//-----------------------------------------------------------------------------
template<class Type>
inline cublasStatus_t mat_vec_mul(cublasOperation_t trans,
                                  const geom_mm<Type> &A, const Type *x,
                                  Type *y, cudaStream_t stream = 0) {
#ifdef PROFILING
	if (trans == CUBLAS_OP_N) {
		profile_info[6].valid = true;
		sprintf(profile_info[6].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 4 );
	} else {
		profile_info[7].valid = true;
		sprintf(profile_info[7].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 7 );
	}
	HANDLE_ERROR(cudaEventRecord(start));
#endif

	int len_y = (trans == CUBLAS_OP_N) ? A.A.numMPs * A.A.numPaths : A.A.rv_x *
	        A.A.rv_y * A.A.rv_z;
	HANDLE_ERROR(cudaMemsetAsync(y, 0, len_y * sizeof(Type), stream));

	// Grid/Block dimensions for dyn-calc multiply
	dim3 threads, blocks;

	if (trans == CUBLAS_OP_N) {
		//emitter  = blockIdx.z * gridDim.y + blockIdx.y;   // 0 .. 627
		//receiver = blockIdx.x * blockDim.x + threadIdx.x;
		threads.x = 256;
		threads.y = 1;
		threads.z = 1;
		blocks.x = 6;
		blocks.y = 4;
		blocks.z = 157;
	} else {
		threads.x = 288; // = 36 * 8 = (4*9) * 8; = 32 * 9 --> evenly dividable by emitter-receiver-number and warp-size
		//                                                         --> for one emitting TAS (4 emitters) and 16 receiving TAS (9 receivers) --> 1*4 * 8*9 = 288
		threads.y = 1;
		threads.z = 1;
		blocks.x = 20;  // 1413 / 72 = 1413 / (8*9) = 19,625 ~ 20 --> 20 of these blocks provide (more than) necessary threads for processing all receiving TAS and one emitting TAS
		//										  --> 20 * 288 = 5760 < 157*9 * 1*4 = 5652 (1,9% overhead)
		blocks.y = 157; // number of emitting TAS --> 157 * 5760 = 904320 > 887364 = 157*4 * 157*9
		blocks.z = 1;

	}

	// Grid/Block dimensions for reduction of partial volumes
	dim3 threads0, blocks0;
	threads0.x = 32 * (A.sub_vol / 2);
	threads0.y = 1;
	threads0.z = 1;
	blocks0.x = A.A.rv_x * (A.sub_vol / 2) / threads0.x;
	blocks0.y = A.A.rv_y / threads0.y;
	blocks0.z = A.A.rv_z / threads0.z;

	unsigned int startPath = 0;

	if ( trans == CUBLAS_OP_N ) {

		for (int i = 0; i < A.A.numMPs; i++) {

			dyn_calc_kernel_wrapper(blocks, threads, 0, stream,
			                        trans, A.sub_vol,
			                        A.A.num_emitters,
			                        A.A.num_receivers, A.A.rv_x,
			                        A.A.rv_y, A.A.rv_z,
			                        (Type)A.A.scale_factor,
			                        A.A.x_re_dev_ptr()[i],
			                        A.A.y_re_dev_ptr()[i],
			                        A.A.z_re_dev_ptr()[i],
			                        A.A.x_em_dev_ptr()[i],
			                        A.A.y_em_dev_ptr()[i],
			                        A.A.z_em_dev_ptr()[i],
			                        x, &(y[startPath]),
			                        A.A.use_path_dev_ptr(), A.params,
			                        0);

			startPath += A.A.numPaths;
		}

	} else {

		HANDLE_ERROR(cudaMemsetAsync(A.y_local, 0, A.sub_vol * len_y *
		                             sizeof(Type), stream));

		for (int i = 0; i < A.A.numMPs; i++) {

			dyn_calc_kernel_wrapper(blocks, threads, 0, stream,
			                        trans, A.sub_vol,
			                        A.A.num_emitters,
			                        A.A.num_receivers, A.A.rv_x,
			                        A.A.rv_y, A.A.rv_z,
			                        (Type)A.A.scale_factor,
			                        A.A.x_re_dev_ptr()[i],
			                        A.A.y_re_dev_ptr()[i],
			                        A.A.z_re_dev_ptr()[i],
			                        A.A.x_em_dev_ptr()[i],
			                        A.A.y_em_dev_ptr()[i],
			                        A.A.z_em_dev_ptr()[i],
			                        &(x[startPath]), A.y_local,
			                        A.A.use_path_dev_ptr(), A.params,
			                        0);

			startPath += A.A.numPaths;

		}

		sum_vol_up_kernel_wrapper(blocks0, threads0,
		                          threads0.x * threads0.y * threads0.z *
		                          sizeof(Type), stream,
		                          A.sub_vol, A.A.rv_x, A.A.rv_y,
		                          A.A.rv_z, y, A.y_local);
	}

#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	if (trans == CUBLAS_OP_N) {
		profile_info[6].time += elapsedTime;
		profile_info[6].runs++;
	} else {
		profile_info[7].time += elapsedTime;
		profile_info[7].runs++;
	}
#endif

	return CUBLAS_STATUS_SUCCESS;

}
#endif

/******************************************************************************
* Single-GPU processing for dynamically calculated matrices
******************************************************************************/
#if (0)
//-----------------------------------------------------------------------------
// matrix-vector multiplication with dynamic calculation of the measurement 
// matrix (kernel moved to separate file)
//-----------------------------------------------------------------------------
template<class Type>
inline cublasStatus_t mat_vec_mul(cublasOperation_t trans,
                                  const geometry_device &A, const Type *x,
                                  Type *y, cudaStream_t stream = 0) {
#ifdef PROFILING
	if (trans == CUBLAS_OP_N) {
		profile_info[6].valid = true;
		sprintf(profile_info[6].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 4 );
	} else {
		profile_info[7].valid = true;
		sprintf(profile_info[7].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 7 );
	}
	HANDLE_ERROR(cudaEventRecord(start));
#endif

	int len_y = (trans == CUBLAS_OP_N) ? A.numMPs * A.num_emitters *
	        A.num_receivers : A.rv_x * A.rv_y * A.rv_z;
	cudaMemset(y, 0, len_y * sizeof(Type));
	dim3 threads, blocks;

	if (trans == CUBLAS_OP_N) {
		threads.x = 512;
		threads.y = 1;
		threads.z = 1;
		blocks.x = 3;
		blocks.y = 4;
		blocks.z = 157;
	} else {
		threads.x = 512;
		threads.y = 1;
		threads.z = 1;
		blocks.x = 3;
		blocks.y = 4;
		blocks.z = 157;
	}

/* thread/block division (requires reverting of changes in kernel files!
        threads.x = 64;
        threads.y = 4;
        if(trans == CUBLAS_OP_N) {
                blocks.x = max(A.num_receivers / threads.x, (unsigned int)1);
                blocks.y = max(A.num_emitters / threads.x, (unsigned int)1);
        } else {
                blocks.y = max(A.num_receivers / threads.x, (unsigned int)1);
                blocks.x = max(A.num_emitters / threads.x, (unsigned int)1);
        }

	// best for v2 volume: 64x64x64
        if(trans == CUBLAS_OP_N) {
                threads.x = 32;
                threads.y = 8;
                blocks.x = 40;
                blocks.y = 78;
        } else {
                threads.x = 128;
                threads.y = 2;
                blocks.x = 1;
                blocks.y =16;
        }*/

	// best for v2 volume: 128x128x128:
/*	if(trans == CUBLAS_OP_N) {
                threads.x = 64;
                threads.y = 4;
                blocks.x = 22;
                blocks.y = 152;
        } else {
                threads.x = 64;
                threads.y = 8;
                blocks.x = 8;
                blocks.y = 26;
        }*/

	unsigned int startPath = 0;

	for (int i = 0; i < A.numMPs; i++) {

		if ( trans == CUBLAS_OP_N ) {

			dyn_calc_kernel_wrapper(blocks, threads, 0, stream,
			                        trans, 1, A.num_emitters,
			                        A.num_receivers, A.rv_x, A.rv_y,
			                        A.rv_z, (Type)A.scale_factor,
			                        A.x_re_dev_ptr()[i],
			                        A.y_re_dev_ptr()[i],
			                        A.z_re_dev_ptr()[i],
			                        A.x_em_dev_ptr()[i],
			                        A.y_em_dev_ptr()[i],
			                        A.z_em_dev_ptr()[i],
			                        x, &(y[startPath]),
			                        A.use_path_dev_ptr());

		} else {

			dyn_calc_kernel_wrapper(blocks, threads, 0, stream,
			                        trans, 1, A.num_emitters,
			                        A.num_receivers, A.rv_x, A.rv_y,
			                        A.rv_z, (Type)A.scale_factor,
			                        A.x_re_dev_ptr()[i],
			                        A.y_re_dev_ptr()[i],
			                        A.z_re_dev_ptr()[i],
			                        A.x_em_dev_ptr()[i],
			                        A.y_em_dev_ptr()[i],
			                        A.z_em_dev_ptr()[i],
			                        &(x[startPath]), y,
			                        A.use_path_dev_ptr());

		}

		startPath += A.num_emitters * A.num_receivers;

	}

#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	if (trans == CUBLAS_OP_N) {
		profile_info[6].time += elapsedTime;
		profile_info[6].runs++;
	} else {
		profile_info[7].time += elapsedTime;
		profile_info[7].runs++;
	}
#endif

	return CUBLAS_STATUS_SUCCESS;

}
#endif

/******************************************************************************
* Multi-GPU processing for dynamically calculated matrices with hierarchical 
* T-multiply
******************************************************************************/
#if (1)

const int emitterPerTas = 4;
const int receiverPerTas = 9;

//-----------------------------------------------------------------------------
// type definition for multi GPU geometry struct and hierarchical T-multiply
//-----------------------------------------------------------------------------
template<class Type>
struct geom_mm_multGPU {
	int mainDevice;
	int numDevices;
	bool mainUsedForSpMV;
	int *deviceArray;
	cudaStream_t *streams;
	int sub_vol;
	geometry_device **A;
	textureParams<Type> **params;
	Type **memA;
	Type **memB;
	Type **memTmp;
	Type  *memAcc;
	const bool allocedInConstructor;

	geom_mm_multGPU(const geometry_host &A_host, int numReqDevices,
	                int *reqDeviceArray, int mainDevice, int sub_vol = 1 );
	geom_mm_multGPU(int mainDevice, int numDevices, bool mainUsedForSpMV,
	                int *deviceArray, cudaStream_t *streams, int sub_vol,
	                geometry_device **A, textureParams<Type> **params,
	                Type **memA, Type **memB, Type **memTmp, Type  *memAcc);
	~geom_mm_multGPU();

};

template<class Type>
geom_mm_multGPU<Type>::geom_mm_multGPU(int mainDevice, int numDevices,
                                       bool mainUsedForSpMV, int *deviceArray,
                                       cudaStream_t *streams, int sub_vol,
                                       geometry_device **A,
                                       textureParams<Type> **params,
                                       Type **memA, Type **memB, Type **memTmp,
                                       Type  *memAcc) :

	mainDevice(mainDevice), numDevices(numDevices), mainUsedForSpMV(
	        mainUsedForSpMV), deviceArray(deviceArray), streams(streams),
	sub_vol(
	        sub_vol),
	A(A), params(params), memA(memA), memB(memB), memTmp(memTmp), memAcc(
	        memAcc), allocedInConstructor(false) {
};

template<class Type>
geom_mm_multGPU<Type>::geom_mm_multGPU(const geometry_host &A_host,
                                       int numReqDevices, int *reqDeviceArray,
                                       int mainDevice, int sub_vol) :
	mainDevice(mainDevice), sub_vol(sub_vol), allocedInConstructor(true) {

	printf(
	        "---------------------------------- geo init ----------------------------------\n");

	mainUsedForSpMV = false;

	// Get names number and names of GPU devices --------------------------------
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
			printf( "Device %s can not be used as it can not access peer memory on " \
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

	// Determine optimal number of signals per submatrix ------------------------
	int pathsMainOpt = 0, pathsOpt = 0;

	double mainOverFactor = 1; //1.25; // must be < max(2, numDevices) TODO: find dependence on numDevices!

	if ((mainUsedForSpMV == true) & (numDevices > 1)) {

		pathsMainOpt = floor(
		        (double)(A_host.numPaths *
		                 A_host.numMPs) /
		        (double)(numDevices + mainOverFactor -
		                 1) * mainOverFactor);
		pathsOpt =
		        ceil( (double)(A_host.numPaths * A_host.numMPs - pathsMainOpt) /
		                (double)(numDevices - 1));

	} else {

		pathsMainOpt =
		        ceil((float)(A_host.numPaths * A_host.numMPs) / (float)numDevices);
		pathsOpt = pathsMainOpt;

	}

	// Generate data elements for sub-geometries  -------------------------------
	int numTotalSignals = A_host.num_emitters * A_host.num_receivers *
	                      A_host.numMPs;
	int numSignalsPerMP = A_host.num_emitters * A_host.num_receivers;

	int numTotalPath = A_host.numPaths * A_host.numMPs;
	int numPathPerMP = A_host.numPaths;

	int currPath = 0, totPath = 0, currSignal = 0;
	int path_tmp = 0, path_val = 0, last_path_val = 0;
	int emitter_tmp = 0, idx_tmp = 0, tas_tmp = 0;

	unsigned int  *num_signals = new unsigned int [numDevices];
	unsigned int  *num_path = new unsigned int [numDevices];

	uint2 *global_path_array = new uint2[numTotalPath];
	uint2 **use_path = new uint2*[numDevices];

	int numActDevices = 0, pathsPerGPUopt = 0;

	if ( deviceArray[numActDevices] == mainDevice )
		pathsPerGPUopt = pathsMainOpt;
	else
		pathsPerGPUopt = pathsOpt;

	int i = 0;
	while (i < numTotalSignals ) {

		if ((currPath >= pathsPerGPUopt) &
		    (i % (emitterPerTas * A_host.num_receivers) == 0)) {

			use_path[numActDevices] = &(global_path_array[path_tmp]);
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

			if ( A_host.use_path[i % numSignalsPerMP] != 0) {

				global_path_array[totPath].x = i /
				                               A_host.
				                               num_receivers;
				global_path_array[totPath].y =
				        (i / (numSignalsPerMP)) * A_host.num_receivers +
				        (i % A_host.num_receivers);

				currPath++;
				totPath++;
			}

			i++;
		}

	}

	if ( currPath > 0) {

		use_path[numActDevices] = &(global_path_array[path_tmp]);

		num_signals[numActDevices] = i - idx_tmp;
		num_path[numActDevices] = currPath;

		numActDevices++;

	}

	numDevices = numActDevices;

	// Distribute geometry to devices -------------------------------------------
	double difftime;
	timeval start, stop;

	gettimeofday(&start, NULL);

	A = new geometry_device*[numDevices];
	params = new textureParams<Type>*[numDevices];
	streams = new cudaStream_t[numDevices];
	memA = new Type*[numDevices];
	memB = new Type*[numDevices];
	memTmp = new Type*[numDevices];

	printf(
	        "numDevices=%i, numTotalSignals=%i, numTotalPaths=%i, numTAS=%i\n",
	        numDevices, numTotalSignals, A_host.numPaths * A_host.numMPs,
	        (numTotalSignals / (emitterPerTas * A_host.num_receivers)));

	for (int i = 0; i < numDevices; i++) {

		HANDLE_ERROR(cudaSetDevice(deviceArray[i]));

		if (  deviceArray[i] != mainDevice )
			HANDLE_ERROR(cudaDeviceReset());

		A[i] = new geometry_device(A_host, (unsigned int*)(use_path[i]),
		                           num_path[i], num_signals[i]);
		params[i] = new textureParams<Type>(A_host.rv_x, A_host.rv_y,
		                                    A_host.rv_z, 0);

		HANDLE_ERROR(cudaMalloc((void**)&(memA[i]), A_host.rv_x *
		                        A_host.rv_y * A_host.rv_z *
		                        sizeof(Type) ));
		HANDLE_ERROR(cudaMalloc((void**)&(memB[i]), num_path[i] *
		                        sizeof(Type) ));

		if ( sub_vol > 1)
			HANDLE_ERROR(cudaMalloc((void**)&(memTmp[i]), sub_vol *
			                        A_host.rv_x * A_host.rv_y *
			                        A_host.rv_z *
			                        sizeof(Type) ));
		else
			memTmp[i] = memA[i];

		HANDLE_ERROR(cudaStreamCreate(&(streams[i])));

		if ( deviceArray[i] != mainDevice )
			HANDLE_ERROR(cudaDeviceEnablePeerAccess(mainDevice, 0));


		printf( "GPU %i: numSignals=%i (%2.1f \%), numPaths=%i (%2.1f \%)\n",
		        deviceArray[i], num_signals[i], num_signals[i] / 
						(float)numTotalSignals * 100, num_path[i], num_path[i] / 
						(float)numTotalPath * 100);

	}

	HANDLE_ERROR(cudaSetDevice(mainDevice));

	if ( numDevices > 1)
		HANDLE_ERROR(cudaMalloc((void**)&(memAcc),
		                        (numDevices - 1) * A_host.rv_x * A_host.rv_y *
		                        A_host.rv_z * sizeof(Type) ));

	else
		memAcc = NULL;

	gettimeofday(&stop, NULL);
	difftime = (double)((stop.tv_sec * 1000000 + stop.tv_usec) - 
							(start.tv_sec * 1000000 + start.tv_usec));
	printf("Time for GPU init: %.3fms\n", difftime / 1000);

	delete(num_signals);
	delete(num_path);
	delete(use_path);
	delete(global_path_array);

};

template<class Type>
geom_mm_multGPU<Type>::~geom_mm_multGPU() {

	if ( allocedInConstructor == true ) {

		for (int i = 0; i < numDevices; i++) {

			HANDLE_ERROR(cudaSetDevice(deviceArray[i]));

			delete(A[i]);
			delete(params[i]);

			if (memA[i] != NULL) cudaFree(memA[i]);

			if (memB[i] != NULL) cudaFree(memA[i]);

			if (memTmp[i] != NULL) cudaFree(memTmp[i]);

			HANDLE_ERROR(cudaStreamDestroy(streams[i]));

		}

		delete(A);
		delete(params);
		delete(memA);
		delete(memB);
		delete(memTmp);

		if (deviceArray != NULL) delete(deviceArray);
		deviceArray = NULL;

		HANDLE_ERROR(cudaSetDevice(mainDevice));

		if (memAcc != NULL) cudaFree(memAcc); memAcc = NULL;

	}

}

//-----------------------------------------------------------------------------
// matrix-vector multiplication with dynamic calculation of the measurement 
// matrix on multiple GPUs
//-----------------------------------------------------------------------------
template<class Type>
inline cublasStatus_t mat_vec_mul(cublasOperation_t trans,
                                  const geom_mm_multGPU<Type> &A, const Type *x,
                                  Type *y, cudaStream_t stream = 0,
                                  bool mulChain = false) {
#ifdef PROFILING

	if (trans == CUBLAS_OP_N) {
		profile_info[6].valid = true;
		sprintf(profile_info[6].name, "%s (N)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 4 );
	} else {
		profile_info[7].valid = true;
		sprintf(profile_info[7].name, "%s (T)\t (%s,%i)\0",
		        __FUNCTION__, __FILE__, __LINE__ - 7 );
	}
	HANDLE_ERROR(cudaEventRecord(start));
#endif

	// Grid/Block dimensions for dyn-calc multiply:
	// CUBLAS_OP_N:
	//		emitter  = blockIdx.z * gridDim.y + blockIdx.y;   // gridDim.z = number of emitting TAS changes for multi-GPU setting
	//		receiver = blockIdx.x * blockDim.x + threadIdx.x;


	dim3 multThreads;

	if (trans == CUBLAS_OP_N)
		multThreads.x = 256;
	else
		multThreads.x = 288; // = 36 * 8 = (4*9) * 8; = 32 * 9 --> evenly dividable by emitter-receiver-number and warp-size
	//                                                         --> for one emitting TAS (4 emitters) and 16 receiving TAS (9 receivers) --> 1*4 * 8*9 = 288

	/*dim3 *multBlocks = new dim3[A.numDevices];
	   for (int i=0; i < A.numDevices; i++) {
	        if(trans == CUBLAS_OP_N) {
	                multBlocks[i].x  = 6;
	                multBlocks[i].y  = 4;
	                multBlocks[i].z  = A.numTAS[i];
	        } else {
	                multBlocks[i].x  = 20;	// 1413 / 72 = 1413 / (8*9) = 19,625 ~ 20 --> 20 of these blocks provide (more than) necessary threads for processing all receiving TAS and one emitting TAS
	                                                                //										  --> 20 * 288 = 5760 < 157*9 * 1*4 = 5652 (1,9% overhead)
	                multBlocks[i].y  = A.numTAS[i]; // number of emitting TAS --> 157 * 5760 = 904320 > 887364 = 157*4 * 157*9 : changes between GPUs
	                multBlocks[i].z  = 1;
	        }
	   }*/

	dim3 *multBlocks = new dim3[A.numDevices];
	for (int i = 0; i < A.numDevices; i++)
		multBlocks[i].x =
		        ceil(A.A[i]->numPaths / (double)multThreads.x);

	// Grid/Block dimensions for reduction of partial volumes
	// intra-reduction: reduction of volumes created by ONE dyn...kernel_wrapper call on a SINGLE GPU
	dim3 intraAccThreads, intraAccBlocks;

	if ( A.sub_vol > 1) {
		intraAccThreads.x = 32 * (A.sub_vol / 2);

		intraAccBlocks.x =
		        (A.A[0]->rv_x * (A.sub_vol / 2)) / intraAccThreads.x;
		intraAccBlocks.y = A.A[0]->rv_y / intraAccThreads.y;
		intraAccBlocks.z = A.A[0]->rv_z / intraAccThreads.z;
	}

	// Grid/Block dimensions for reduction of partial volumes
	// inter-reduction: reduction of volumes created by ALL dyn...kernel_wrapper calls on a MULTIPLE GPUs after collection on mainDevice
	dim3 interAccThreads, interAccBlocks;

	if ( A.numDevices > 1 ) {
		interAccThreads.x = 32;

		interAccBlocks.x = A.A[0]->rv_x / interAccThreads.x;
		interAccBlocks.y = A.A[0]->rv_y / interAccThreads.y;
		interAccBlocks.z = A.A[0]->rv_z / interAccThreads.z;
	}

	const Type **x_local = new const Type*[A.numDevices];
	Type **y_local = new Type*[A.numDevices];

	const geometry_device *mA;

	unsigned int device = 0;
	unsigned int currPos = 0;
	unsigned int volSize = A.A[0]->rv_x * A.A[0]->rv_y * A.A[0]->rv_z;;

	cudaEvent_t inputStreamFinalize;
	HANDLE_ERROR(cudaEventCreate(&inputStreamFinalize));
	HANDLE_ERROR(cudaEventRecord(inputStreamFinalize, stream));

	cudaEvent_t *calcCompEvent = new cudaEvent_t[A.numDevices];

	for (int i = 0; i < A.numDevices; i++ ) {

		HANDLE_ERROR(cudaSetDevice(A.deviceArray[i]));
		HANDLE_ERROR(cudaEventCreate(&(calcCompEvent[i])));

		HANDLE_ERROR(cudaStreamWaitEvent(A.streams[i],
		                                 inputStreamFinalize, 0));

		mA = A.A[i];

		volSize = mA->rv_x * mA->rv_y * mA->rv_z;

		// --- CUBLAS_OP_N ---
		if ( trans == CUBLAS_OP_N ) {

			// Check if current device is main device
			if ( A.deviceArray[i] != A.mainDevice ) {

				// Set local pointers to pre-alloced memory
				x_local[i] = x;
				y_local[i] = A.memB[i];

			} else {

				x_local[i] = x;
				y_local[i] = &(y[currPos]);

			}

			// Broadcast vector x: copy complete vector from input array to 
			// 3D-alloced array for later copy to array (main device and all further 
			// devices)
			HANDLE_ERROR(cudaMemcpy2DAsync(A.params[i]->
			                               pitchedDevPtr.ptr,
			                               A.params[i]->
			                               pitchedDevPtr.pitch, x,
			                               mA->rv_y * sizeof(Type),
			                               mA->rv_y * sizeof(Type),
			                               mA->rv_x * mA->rv_z,
			                               cudaMemcpyDeviceToDevice,
			                               A.streams[i]));

			// Do dynamic SpMV on sub-matrices
			dyn_calc_kernel_wrapper(multBlocks[i], multThreads, 0,
			                        A.streams[i],
			                        trans, A.sub_vol,
			                        mA->num_emitters,
			                        mA->num_receivers, mA->rv_x,
			                        mA->rv_y, mA->rv_z,
			                        (Type)mA->scale_factor,
			                        mA->x_re_dev_ptr(),
			                        mA->y_re_dev_ptr(),
			                        mA->z_re_dev_ptr(),
			                        mA->x_em_dev_ptr(),
			                        mA->y_em_dev_ptr(),
			                        mA->z_em_dev_ptr(),
			                        x_local[i], y_local[i],
			                        mA->use_path_dev_ptr(),
			                        A.params[i],
			                        mA->numPaths);

			// Collect sub-results: Copy vector y back to main device
			if ( A.deviceArray[i] != A.mainDevice )
				HANDLE_ERROR(cudaMemcpyAsync((void*)(&y[currPos]),
				                             y_local[i],
				                             mA->numPaths *
				                             sizeof(Type),
				                             cudaMemcpyDeviceToDevice,
				                             A.streams[i]));

			// --- CUBLAS_OP_T ---
		} else { 

			// Check if current device is main device: If so, no memory transfer 
			// necessary!
			if ( A.deviceArray[i] != A.mainDevice ) {

				// Set local pointers to pre-alloced memory
				x_local[i] = A.memB[i];
				y_local[i] = A.memA[i];

				// Broadcast vector x: copy complete vector from main device to all 
				// devices
				if ( mulChain == false )
					HANDLE_ERROR(cudaMemcpyAsync((void*)( x_local [i]), 
								(void*)(&x[currPos]), mA->numPaths * sizeof(Type),
								cudaMemcpyDeviceToDevice, A.streams[i]));
			} else {

				x_local[i] = &(x[currPos]);
				y_local[i] = y;

			}

			// Do dynamic SpMV on sub matrices
			HANDLE_ERROR(cudaMemsetAsync(y_local[i], 0, volSize * sizeof(Type), 
						A.streams[i]));

			if ( A.sub_vol > 0)
				HANDLE_ERROR(cudaMemsetAsync(A.memTmp[i], 0, 
							A.sub_vol * volSize * sizeof(Type), A.streams[i]));

			dyn_calc_kernel_wrapper(multBlocks[i], multThreads, 0,
			                        A.streams[i], trans, A.sub_vol,
			                        mA->num_emitters,
			                        mA->num_receivers, mA->rv_x,
			                        mA->rv_y, mA->rv_z,
			                        (Type)mA->scale_factor,
			                        mA->x_re_dev_ptr(),
			                        mA->y_re_dev_ptr(),
			                        mA->z_re_dev_ptr(),
			                        mA->x_em_dev_ptr(),
			                        mA->y_em_dev_ptr(),
			                        mA->z_em_dev_ptr(),
			                        x_local[i], A.memTmp[i],
			                        mA->use_path_dev_ptr(),
			                        A.params[i],
			                        mA->numPaths);

			if ( A.sub_vol > 1)
				sum_vol_up_kernel_wrapper(intraAccBlocks,
				                          intraAccThreads,
				                          intraAccThreads.x * intraAccThreads.y * intraAccThreads.z *
				                          sizeof(Type),
				                          A.streams[i],
				                          A.sub_vol, mA->rv_x,
				                          mA->rv_y, mA->rv_z,
				                          y_local[i],
				                          A.memTmp[i]);
			else if (A.deviceArray[i] == A.mainDevice)
				HANDLE_ERROR(cudaMemcpyAsync((void*)(y_local[i]),
				                             A.memTmp[i],
				                             volSize *
				                             sizeof(Type),
				                             cudaMemcpyDeviceToDevice,
				                             A.streams[i]));

			// Collect sub-results: Copy vector y back to main device if on more than one device
			if ( A.deviceArray[i] != A.mainDevice  ) {

				HANDLE_ERROR(cudaMemcpyAsync((void*)(&A.memAcc[ device * volSize]),
				                             y_local[i], volSize * sizeof(Type),
				                             cudaMemcpyDeviceToDevice,
				                             A.streams[i]));
				device++;
			}

		}

		currPos += mA->numPaths;

		HANDLE_ERROR(cudaEventRecord(calcCompEvent[i], A.streams[i]));

	}

	HANDLE_ERROR(cudaSetDevice(A.mainDevice));

	for (int i = 0; i < A.numDevices; i++) {

		HANDLE_ERROR(cudaStreamWaitEvent(stream, calcCompEvent[i], 0));

		HANDLE_ERROR(cudaEventDestroy(calcCompEvent[i]));

	}

	if ((trans == CUBLAS_OP_T) & (A.numDevices > 1)) {

		sum_vol_up_kernel_inter_wrapper(interAccBlocks, interAccThreads,
		                                interAccThreads.x * interAccThreads.y * interAccThreads.z *
		                                sizeof(Type), stream,
		                                A.numDevices - 1, true,
		                                mA->rv_x, mA->rv_y, mA->rv_z, y,
		                                A.memAcc);

	}

#ifdef PROFILING
	HANDLE_ERROR(cudaEventRecord(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	if (trans == CUBLAS_OP_N) {
		profile_info[6].time += elapsedTime;
		profile_info[6].runs++;
	} else {
		profile_info[7].time += elapsedTime;
		profile_info[7].runs++;
	}
#endif

	return CUBLAS_STATUS_SUCCESS;

}
#endif

#endif /* MAT_VEC_MUL_DYNAMIC_H_ */
