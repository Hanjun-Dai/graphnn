#include "gpuhandle.h"

__global__ void SetupRandKernel(curandState_t *state, unsigned long long seed) 
{
    const unsigned int tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, tidx, 0, &state[tidx]);
}

void GPUHandle::Init(int dev_id, unsigned int _streamcnt)
{
	streamcnt = _streamcnt;
	cudaDeviceReset();
	cudaSetDevice(dev_id);
	streams = new cudaStream_t[streamcnt];
	for(unsigned int id = 0; id < streamcnt; ++id)
	{
		cudaStreamCreate(&streams[id]);
	}
	cublasCreate(&cublashandle);
	cusparseCreate(&cusparsehandle);
	curandCreateGenerator(&curandgenerator, CURAND_RNG_PSEUDO_DEFAULT);
	
	curandSetPseudoRandomGeneratorSeed(curandgenerator, time(NULL));
	
    cudaMalloc((void **)&devRandStates, NUM_RND_STREAMS * sizeof(curandState_t));
	SetupRandKernel<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(devRandStates, 1 + time(NULL)*2);
}

void GPUHandle::Destroy()
{
	for(unsigned int id = 0; id < streamcnt; ++id)
	{
		cudaStreamDestroy(streams[id]);
	}
	cublasDestroy_v2(cublashandle);
	cusparseDestroy(cusparsehandle);
	curandDestroyGenerator(curandgenerator);
    cudaFree(devRandStates);
	streamcnt = 0U;
}

curandState_t* GPUHandle::devRandStates = NULL;
cudaStream_t* GPUHandle::streams = NULL;
cublasHandle_t GPUHandle::cublashandle;
cusparseHandle_t GPUHandle::cusparsehandle;
curandGenerator_t GPUHandle::curandgenerator;
unsigned int GPUHandle::streamcnt = 1U;
