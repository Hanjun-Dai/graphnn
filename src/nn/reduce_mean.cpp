#include "nn/reduce_mean.h"

namespace gnn
{


template<typename mode, typename Dtype>
ReduceMean<mode, Dtype>::ReduceMean(std::string _name, PropErr _properr) 
				: Factor(_name, _properr)
{

}

template class ReduceMean<CPU, float>;
template class ReduceMean<CPU, double>;


}