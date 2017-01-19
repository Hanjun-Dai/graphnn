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

template<typename mode, typename Dtype>
void ParamSet<mode, Dtype>::Save(std::string filename)
{
	FILE* fid = fopen(filename.c_str(), "wb");

	for (auto& p : params)
	{
		p.second->Serialize(fid);
	}

	fclose(fid);
}

template<typename mode, typename Dtype>
void ParamSet<mode, Dtype>::Load(std::string filename)
{
	FILE* fid = fopen(filename.c_str(), "rb");

	for (auto& p : params)
		p.second->Deserialize(fid);

	fclose(fid);
}

template class ParamSet<CPU, float>;
template class ParamSet<CPU, double>;
template class ParamSet<GPU, float>;
template class ParamSet<GPU, double>;


}