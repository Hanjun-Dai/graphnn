#include "nn/relu.h"

namespace gnn
{

template<typename mode, typename Dtype>
ReLU<mode, Dtype>::ReLU(std::string _name, PropErr _properr) : Factor(_name, _properr)
{

}

template class ReLU<CPU, float>;
template class ReLU<CPU, double>;

}