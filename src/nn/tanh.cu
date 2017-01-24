#include "nn/tanh.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_unary_functor.h"

namespace gnn
{

template<typename Dtype>
void TanhAct(DTensor<GPU, Dtype>& in, DTensor<GPU, Dtype>& out)
{
	out.CopyFrom(in);
	UnaryEngine<GPU>::Exec<UnaryTanh>(out.data->ptr, out.shape.Count());
}

template void TanhAct(DTensor<GPU, float>& in, DTensor<GPU, float>& out);
template void TanhAct(DTensor<GPU, double>& in, DTensor<GPU, double>& out);

template<typename Dtype>
__global__ void TanhDerivKernel(Dtype *dst, Dtype *out, Dtype* cur_grad, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        dst[i] += cur_grad[i] * (1 - out[i] * out[i]);
    }
}

template<typename Dtype>
void TanhDeriv(DTensor<GPU, Dtype>& dst, DTensor<GPU, Dtype>& cur_output, DTensor<GPU, Dtype>& cur_grad)
{
	int thread_num = c_uCudaThreadNum;
	if (dst.shape.Count() < thread_num)
		thread_num = dst.shape.Count();
    int blocksPerGrid = (dst.shape.Count() + thread_num - 1) / thread_num;
    TanhDerivKernel <<< blocksPerGrid, thread_num, 0, cudaStreamPerThread >>>(dst.data->ptr, cur_output.data->ptr, cur_grad.data->ptr, dst.shape.Count());
}

template void TanhDeriv(DTensor<GPU, float>& dst, DTensor<GPU, float>& cur_output, DTensor<GPU, float>& cur_grad);
template void TanhDeriv(DTensor<GPU, double>& dst, DTensor<GPU, double>& cur_output, DTensor<GPU, double>& cur_grad);


}