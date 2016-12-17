#ifndef PARAM_SET_H
#define PARAM_SET_H

#include <string>
#include <map>
#include "nn/variable.h"

namespace gnn
{

class ParamSet
{
public:
	ParamSet();

	void AddParam(std::shared_ptr<DiffVar> param);

	std::map<std::string, std::shared_ptr<DiffVar> > params;
};

template<typename ParamType, typename... Args>
std::shared_ptr<ParamType> add_diff(ParamSet& pset, std::string param_name, Args&&... args)
{
	auto p = std::make_shared<ParamType>(param_name, std::forward<Args>(args)...);
	pset.AddParam(p);
	return p;
}

template<typename ParamType, typename IdxType, typename... Args>
std::shared_ptr<ParamType> add_diff(ParamSet& pset, std::string param_name, 
								std::initializer_list<IdxType> l, Args&&... args)
{
	auto p = std::make_shared<ParamType>(param_name, l, std::forward<Args>(args)...);
	pset.AddParam(p);
	return p;
}

}

#endif