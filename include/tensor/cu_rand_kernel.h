#ifndef CUDA_RAND_KERNEL_CUH
#define CUDA_RAND_KERNEL_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "gpu_handle.h"

namespace gnn
{

template<typename Dtype>
class NormalRandomizer
{
public:
	NormalRandomizer(Dtype _mean, Dtype _std) : mean(_mean), std(_std) {}
	__device__ inline Dtype operator()(curandState_t* state)
	{
        return curand_normal(state) * std + mean;
	}
    
private:    
    Dtype mean;
    Dtype std;
};

template<typename Dtype>
class BinomialRandomizer
{
public:
	BinomialRandomizer() {}
	__device__ inline Dtype operator()(curandState_t* state)
	{
		return curand_uniform(state) > 0.5 ? 1.0 : -1.0;
	}
};

template<typename Dtype>
class UniformRandomizer
{
public:
	UniformRandomizer(Dtype _lb, Dtype _ub) : lb(_lb), ub(_ub) {}
	__device__ inline Dtype operator()(curandState_t* state)
	{
		return curand_uniform(state) * (ub - lb) + lb;
	}
    
private:    
    Dtype lb;
    Dtype ub; 
};


template<typename Dtype>
class ChisquareRandomizer
{
public:
	ChisquareRandomizer(Dtype _degree) : alpha(_degree / 2) {}
	
	__device__ inline Dtype operator()(curandState_t* state)
	{
		Dtype x, v, u;
      	Dtype d = alpha - 1.0 / 3.0;
      	Dtype c = (1.0 / 3.0) / sqrt (d);

      	while (1){
       		do {
            	x = curand_normal(state);
              	v = 1.0 + c * x;
          	} while (v <= 0);

          	v = v * v * v;
          	u = curand_uniform(state);

          	if (u < 1 - 0.0331 * x * x * x * x) 
            	break;

          	if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
            	break;
      	}
      	// scale by 2.0 to get chisquare
      	return 2.0 * (d * v);		
	}
	
private:
	const Dtype alpha;
};

template<typename Dtype, class RandEngine>
__global__ void RandKernel(Dtype *targets, int numElements, curandState_t* state, RandEngine rnd)
{
    const int tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    curandState_t localState = state[tidx];
    for (int i = tidx; i < numElements; i += NUM_RND_STREAMS) 
    {
        targets[i] = rnd(&localState);
    }
    state[tidx] = localState;
}

template<typename Dtype, class RandEngine>
void SetRand(Dtype *dst, int numElements, RandEngine rnd)
{
  GpuHandle::rand_lock.lock();
  RandKernel<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(dst, numElements, GpuHandle::devRandStates, rnd);
  GpuHandle::rand_lock.unlock();
}

}

#endif