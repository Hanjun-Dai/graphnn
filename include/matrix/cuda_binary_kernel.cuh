#ifndef CUDA_BINARY_KERNEL_CUH
#define CUDA_BINARY_KERNEL_CUH

#include <cuda_runtime.h>
#include "gpuhandle.h"

//=================================== mul ======================================

template<typename Dtype>
class BinaryMul
{
public:
	BinaryMul() {}
    
	__device__ inline void operator()(Dtype& dst, const Dtype& lhs)
	{
		dst *= lhs;
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& lhs, const Dtype& rhs)
	{
		dst = lhs * rhs;
	}    
};

//=================================== mul ======================================

template<typename Dtype>
class BinaryDiv
{
public:
	BinaryDiv() {}
    
	__device__ inline void operator()(Dtype& dst, const Dtype& lhs)
	{
		dst /= lhs;
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& lhs, const Dtype& rhs)
	{
		dst = lhs / rhs;
	}    
};


//=================================== call interface ======================================

template<typename Dtype, class BinaryEngine>
__global__ void BinaryKernel(Dtype *dst, const Dtype *lhs, int numElements, BinaryEngine binary)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < numElements)
    {
        binary(dst[i], lhs[i]);
    }
}

template<typename Dtype, class BinaryEngine>
void BinaryOp(Dtype *dst, const Dtype *lhs, int numElements, BinaryEngine binary, const unsigned& sid)
{
    int thread_num = min(c_uCudaThreadNum, numElements);    
    int blocksPerGrid = (numElements + thread_num - 1) / thread_num;
    BinaryKernel<<<blocksPerGrid, thread_num, 0, GPUHandle::streams[sid]>>> (dst, lhs, numElements, binary);
}

template<typename Dtype, class BinaryEngine>
__global__ void BinaryKernel(Dtype *dst, const Dtype* lhs, const Dtype *rhs, int numElements, BinaryEngine binary)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < numElements)
    {
        binary(dst[i], lhs[i], rhs[i]);
    }
}

template<typename Dtype, class BinaryEngine>
void BinaryOp(Dtype *dst, const Dtype* lhs, const Dtype *rhs, int numElements, BinaryEngine binary, const unsigned& sid)
{
    int thread_num = min(c_uCudaThreadNum, numElements);    
    int blocksPerGrid = (numElements + thread_num - 1) / thread_num;
    BinaryKernel<<<blocksPerGrid, thread_num, 0, GPUHandle::streams[sid]>>> (dst, lhs, rhs, numElements, binary);
}

#endif