#ifndef GPUHANDLE_H
#define GPUHANDLE_H

#include "mat_typedef.h"
#include <curand.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <ctime>
#include <curand_kernel.h>

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

struct GPUHandle
{
	static cudaStream_t* streams;
	static cublasHandle_t cublashandle;
	static cusparseHandle_t cusparsehandle;
	static curandGenerator_t curandgenerator;
	static unsigned int streamcnt;
	
	static void Init(int dev_id, unsigned int _streamcnt = 1U);
	static void Destroy();
	
	static curandState_t* devRandStates;
};

#endif
