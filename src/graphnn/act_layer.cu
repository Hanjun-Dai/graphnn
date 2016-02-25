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
__global__ void ReLUDerivKernel(Dtype *d, Dtype *c, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        d[i] = c[i] > 0 ? d[i] : 0;
    }
}

template<typename Dtype>
void ReLULayer<GPU, Dtype>::Derivative(DenseMat<GPU, Dtype>& dst, DenseMat<GPU, Dtype>& prev_output, 
                            DenseMat<GPU, Dtype>& cur_output, DenseMat<GPU, Dtype>& cur_grad)
{
    dst.CopyFrom(cur_grad);
    int thread_num = min(c_uCudaThreadNum, dst.count);    
    int blocksPerGrid = (dst.count + thread_num - 1) / thread_num;
    ReLUDerivKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[dst.streamid] >>>(dst.data, cur_output.data, dst.count);
}

template class ReLULayer<GPU, float>;
template class ReLULayer<GPU, double>;