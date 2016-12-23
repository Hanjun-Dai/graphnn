#ifndef GPU_HANDLE_H
#define GPU_HANDLE_H

#include <curand.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <ctime>
#include <curand_kernel.h>
#include <mutex>
#include <thread>
#include <queue>

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

struct GpuContext
{
	int id;
	cublasHandle_t cublashandle;
	cusparseHandle_t cusparsehandle;
	GpuContext(int _id, cublasHandle_t _cublas, cusparseHandle_t _cusparse)
		: id(_id), cublashandle(_cublas), cusparsehandle(_cusparse) {}
};

struct GpuHandle
{
	static curandGenerator_t curandgenerator;
	static unsigned int streamcnt;
	
	static void Init(int dev_id, unsigned int _streamcnt = 1U);
	static void Destroy();
	
	static GpuContext AquireCtx();
	static void ReleaseCtx(const GpuContext& ctx);

	static curandState_t* devRandStates;

private:
	static cublasHandle_t* cublashandles;
	static cusparseHandle_t* cusparsehandles;	
	static std::queue< int > resources;
	static std::mutex r_loc;
	static bool* inUse;
};

#endif