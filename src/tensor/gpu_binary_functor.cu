#include "tensor/gpu_dense_tensor.h"
#define min(x, y) (x < y ? x : y)

namespace gnn
{

template<typename dstType, typename srcType>
__global__ void TypeCastCopyKernel(dstType* dst, srcType* src, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < numElements)
    {
    	dst[i] = (dstType)src[i];
    }
}

template<typename dstType, typename srcType>
void TypeCastCopy(dstType* dst, srcType* src, size_t count)
{
    int thread_num = min(c_uCudaThreadNum, count);    
    int blocksPerGrid = (count + thread_num - 1) / thread_num;
    TypeCastCopyKernel<<<blocksPerGrid, thread_num, 0, cudaStreamPerThread>>> (dst, src, count);
}

template void TypeCastCopy(float* dst, double* src, size_t count);
template void TypeCastCopy(float* dst, int* src, size_t count);
template void TypeCastCopy(double* dst, float* src, size_t count);
template void TypeCastCopy(double* dst, int* src, size_t count);
template void TypeCastCopy(int* dst, float* src, size_t count);
template void TypeCastCopy(int* dst, double* src, size_t count);

}