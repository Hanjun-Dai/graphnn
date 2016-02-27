#include "relu_layer.h"
#include "dense_matrix.h"
#include "cuda_helper.h"
#include "cuda_unary_kernel.cuh"
#include "sparse_matrix.h"
#include <cuda_runtime.h>

#define min(x, y) (x < y ? x : y)

// =========================================== relu layer ================================================
template<typename Dtype>
void ReLULayer<GPU, Dtype>::Act(DenseMat<GPU, Dtype>& prev_out, DenseMat<GPU, Dtype>& cur_out)
{
    UnaryOp(cur_out.data, prev_out.data, prev_out.count, UnaryReLU<Dtype>(), cur_out.streamid);
}

template<typename Dtype>
__global__ void ReLUDerivKernel(Dtype *dst, Dtype *out, Dtype* cur_grad, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements && out[i] > 0)
    {
        dst[i] += cur_grad[i];
    }
}

template<typename Dtype>
void ReLULayer<GPU, Dtype>::Derivative(DenseMat<GPU, Dtype>& dst, DenseMat<GPU, Dtype>& prev_output, 
                            DenseMat<GPU, Dtype>& cur_output, DenseMat<GPU, Dtype>& cur_grad, Dtype beta)
{
    dst.Scale(beta);
    
    int thread_num = min(c_uCudaThreadNum, dst.count);    
    int blocksPerGrid = (dst.count + thread_num - 1) / thread_num;
    ReLUDerivKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[dst.streamid] >>>(dst.data, cur_output.data, cur_grad.data, dst.count);
}

template class ReLULayer<GPU, float>;
template class ReLULayer<GPU, double>;