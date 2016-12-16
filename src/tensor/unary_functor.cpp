#include "tensor/unary_functor.h"

namespace gnn
{

template<typename Dtype>
UnarySet<CPU, Dtype>::UnarySet(Dtype _scalar) : scalar(_scalar) {}

template<typename Dtype>
void UnarySet<CPU, Dtype>::operator()(Dtype& dst)
{
	dst = scalar;
}

template class UnarySet<CPU, float>;
template class UnarySet<CPU, double>;
template class UnarySet<CPU, int>;

}