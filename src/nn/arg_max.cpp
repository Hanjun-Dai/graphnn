#include "nn/arg_max.h"

namespace gnn
{


template<typename mode, typename Dtype>
ArgMax<mode, Dtype>::ArgMax(std::string _name, PropErr _properr) 
				: Factor(_name, _properr)
{

}

template class ArgMax<CPU, float>;
template class ArgMax<CPU, double>;


}