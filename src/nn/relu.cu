#include "nn/relu.h"

namespace gnn
{

template<typename Dtype>
void ReLUAct(DTensor<GPU, Dtype>& in, DTensor<GPU, Dtype>& out)
{

}

template void ReLUAct<float>(DTensor<GPU, float>& in, DTensor<GPU, float>& out);
template void ReLUAct<double>(DTensor<GPU, double>& in, DTensor<GPU, double>& out);

}