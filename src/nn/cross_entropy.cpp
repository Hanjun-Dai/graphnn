#include "nn/cross_entropy.h"

namespace gnn
{


template<typename mode, typename Dtype>
CrossEntropy<mode, Dtype>::CrossEntropy(std::string _name, bool _need_softmax, PropErr _properr) 
				: Factor(_name, _properr), need_softmax(_need_softmax)
{

}

template class CrossEntropy<CPU, float>;
template class CrossEntropy<CPU, double>;


}