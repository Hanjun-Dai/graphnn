#ifndef GPU_HANDLE_H
#define GPU_HANDLE_H

#ifdef USE_GPU

#include <curand.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <ctime>
#include <curand_kernel.h>
#include <mutex>
#include <thread>
#include <queue>

namespace gnn
{

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)
#define c_uCudaThreadNum 1024

#define WITH_GPUCTX(ctx, ...) \
	auto ctx = GpuHandle::AquireCtx(); \
	{__VA_ARGS__} \
	GpuHandle::ReleaseCtx(ctx); \

struct GpuContext
{
	int id;
	cublasHandle_t cublasHandle;
	cusparseHandle_t cusparseHandle;
	GpuContext(int _id, cublasHandle_t _cublas, cusparseHandle_t _cusparse)
		: id(_id), cublasHandle(_cublas), cusparseHandle(_cusparse) {}
};

struct GpuHandle
{
	static curandGenerator_t curandgenerator;
	static unsigned int streamcnt;
	static cudaStream_t cudaRandStream;

	static void Init(int dev_id, unsigned int _streamcnt = 1U);
	static void Destroy();
	
	static GpuContext AquireCtx();
	static void ReleaseCtx(const GpuContext& ctx);

	static curandState_t* devRandStates;
	static std::mutex rand_lock;
	
private:
	static cublasHandle_t* cublashandles;
	static cusparseHandle_t* cusparsehandles;	
	static std::queue< int > resources;
	static std::mutex r_loc;
	static bool* inUse;
};

}

#else

namespace gnn
{

struct GpuHandle
{
	static void Init(int dev_id, unsigned int _streamcnt = 1U) {}
	static void Destroy() {}
};

}
#endif

#endif