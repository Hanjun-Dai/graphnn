#include "tensor/unary_functor.h"

namespace gnn
{

UnarySet<CPU>::UnarySet(Dtype _scalar) : scalar(_scalar) {}

void UnarySet<CPU>::operator()(Dtype& dst)
{
	dst = scalar;
}

}