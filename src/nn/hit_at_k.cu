#include "nn/hit_at_k.h"

namespace gnn
{

template<typename Dtype>
void HitsTopK(DTensor<GPU, Dtype>& pred, SpTensor<GPU, Dtype>& label, DTensor<GPU, int>& out, int k)
{
	ASSERT(false, "not implemented");
}

template void HitsTopK(DTensor<GPU, float>& pred, SpTensor<GPU, float>& label, DTensor<GPU, int>& out, int k);
template void HitsTopK(DTensor<GPU, double>& pred, SpTensor<GPU, double>& label, DTensor<GPU, int>& out, int k);

template class HitAtK<GPU, float>;
template class HitAtK<GPU, double>;

}