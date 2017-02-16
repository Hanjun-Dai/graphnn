#include "nn/jagged_softmax.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_unary_functor.h"

namespace gnn
{

template<typename Dtype>
void JaggedSoftmaxDeriv(DTensor<GPU, Dtype>& dst, DTensor<GPU, Dtype>& cur_output, DTensor<GPU, Dtype>& cur_grad, DTensor<GPU, int>& lens)
{
	
}

template void JaggedSoftmaxDeriv(DTensor<GPU, float>& dst, DTensor<GPU, float>& cur_output, DTensor<GPU, float>& cur_grad, DTensor<GPU, int>& lens);
template void JaggedSoftmaxDeriv(DTensor<GPU, double>& dst, DTensor<GPU, double>& cur_output, DTensor<GPU, double>& cur_grad, DTensor<GPU, int>& lens);


}