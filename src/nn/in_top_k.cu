#include "nn/in_top_k.h"

namespace gnn
{

template<typename Dtype>
void IsInTopK(DTensor<GPU, Dtype>& pred, DTensor<GPU, int>& label, DTensor<GPU, int>& out, int k)
{
	ASSERT(false, "not implemented");
}

template void IsInTopK(DTensor<GPU, float>& pred, DTensor<GPU, int>& label, DTensor<GPU, int>& out, int k);
template void IsInTopK(DTensor<GPU, double>& pred, DTensor<GPU, int>& label, DTensor<GPU, int>& out, int k);

template class InTopK<GPU, float>;
template class InTopK<GPU, double>;
}