#include "nn/type_cast.h"

namespace gnn
{

template<typename mode, typename Dtype>
TypeCast<mode, Dtype>::TypeCast(std::string _name, PropErr _properr) 
				: Factor(_name, _properr)
{

}

template class TypeCast<CPU, float>;
template class TypeCast<CPU, double>;

}