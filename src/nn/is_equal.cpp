#include "nn/is_equal.h"

namespace gnn
{

template<typename mode, typename Dtype>
IsEqual<mode, Dtype>::IsEqual(std::string _name) 
				: Factor(_name, PropErr::N)
{

}

template class IsEqual<CPU, float>;
template class IsEqual<CPU, double>;
template class IsEqual<CPU, int>;

}