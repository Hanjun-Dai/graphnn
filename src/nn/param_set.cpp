#include "nn/param_set.h"
#include <cassert>

namespace gnn
{

template<typename mode, typename Dtype>
ParamSet<mode, Dtype>::ParamSet()
{
	params.clear();
}

template<typename mode, typename Dtype>
void ParamSet<mode, Dtype>::AddParam(std::shared_ptr< DTensorVar<mode, Dtype> > param)
{
	assert(params.count(param->name) == 0);
	params[param->name] = param;
}

template class ParamSet<CPU, float>;
template class ParamSet<CPU, double>;
template class ParamSet<GPU, float>;
template class ParamSet<GPU, double>;


}