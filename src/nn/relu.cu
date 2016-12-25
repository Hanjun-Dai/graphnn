#include "nn/relu.h"

namespace gnn
{

template<typename Dtype>
void ReLUAct(DTensor<GPU, Dtype>& in, DTensor<GPU, Dtype>& out)
{
	ASSERT(false, "not implemented");
}

template void ReLUAct(DTensor<GPU, float>& in, DTensor<GPU, float>& out);
template void ReLUAct(DTensor<GPU, double>& in, DTensor<GPU, double>& out);

template<typename Dtype>
void ReLUDeriv(DTensor<GPU, Dtype>& dst, DTensor<GPU, Dtype>& cur_output, DTensor<GPU, Dtype>& cur_grad)
{
	ASSERT(false, "not implemented");
}

template void ReLUDeriv(DTensor<GPU, float>& dst, DTensor<GPU, float>& cur_output, DTensor<GPU, float>& cur_grad);
template void ReLUDeriv(DTensor<GPU, double>& dst, DTensor<GPU, double>& cur_output, DTensor<GPU, double>& cur_grad);

template class ReLU<GPU, float>;
template class ReLU<GPU, double>;

}