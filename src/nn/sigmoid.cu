#include "nn/sigmoid.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_unary_functor.h"

namespace gnn
{

template<typename Dtype>
__global__ void SigmDerivKernel(Dtype *dst, Dtype *out, Dtype* cur_grad, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        dst[i] += cur_grad[i] * out[i] * (1 - out[i]);
    }
}

template<typename Dtype>
void SigmDeriv(DTensor<GPU, Dtype>& dst, DTensor<GPU, Dtype>& cur_output, DTensor<GPU, Dtype>& cur_grad)
{
	int thread_num = c_uCudaThreadNum;
	if (dst.shape.Count() < thread_num)
		thread_num = dst.shape.Count();
    int blocksPerGrid = (dst.shape.Count() + thread_num - 1) / thread_num;
    SigmDerivKernel <<< blocksPerGrid, thread_num, 0, cudaStreamPerThread >>>(dst.data->ptr, cur_output.data->ptr, cur_grad.data->ptr, dst.shape.Count());
}

template void SigmDeriv(DTensor<GPU, float>& dst, DTensor<GPU, float>& cur_output, DTensor<GPU, float>& cur_grad);
template void SigmDeriv(DTensor<GPU, double>& dst, DTensor<GPU, double>& cur_output, DTensor<GPU, double>& cur_grad);


}