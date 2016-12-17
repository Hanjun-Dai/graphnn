#include "nn/param_set.h"
#include <cassert>

namespace gnn
{

ParamSet::ParamSet()
{
	params.clear();
}

void ParamSet::AddParam(std::shared_ptr<DiffVar> param)
{
	assert(params.count(param->name) == 0);
	params[param->name] = param;
}

}