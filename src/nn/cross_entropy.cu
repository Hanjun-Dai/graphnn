#include "nn/cross_entropy.h"

namespace gnn
{

template<typename Dtype>
void CalcCrossEntropy(DTensor<GPU, Dtype>& prob, SpTensor<GPU, Dtype>& label, DTensor<GPU, Dtype>& out)
{
	ASSERT(false, "not implemented");
}

template void CalcCrossEntropy(DTensor<GPU, float>& prob, SpTensor<GPU, float>& label, DTensor<GPU, float>& out);
template void CalcCrossEntropy(DTensor<GPU, double>& prob, SpTensor<GPU, double>& label, DTensor<GPU, double>& out);

template class CrossEntropy<GPU, float>;
template class CrossEntropy<GPU, double>;

}