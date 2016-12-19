#ifndef PARAM_SET_H
#define PARAM_SET_H

#include <string>
#include <map>
#include "nn/variable.h"

namespace gnn
{

template<typename mode, typename Dtype>
class ParamSet
{
public:
	ParamSet();

	void AddParam(std::shared_ptr< DTensorVar<mode, Dtype> > param);

	std::map<std::string, std::shared_ptr< DTensorVar<mode, Dtype> > > params;
};

template<template <typename, typename> class ParamType, typename mode, typename Dtype, typename... Args>
std::shared_ptr< ParamType<mode, Dtype> > add_diff(ParamSet<mode, Dtype>& pset, std::string param_name, Args&&... args)
{
	auto p = std::make_shared< ParamType<mode, Dtype> >(param_name, std::forward<Args>(args)...);
	pset.AddParam(p);
	return p;
}

template<template <typename, typename> class ParamType, typename mode, typename Dtype, typename IdxType, typename... Args>
std::shared_ptr< ParamType<mode, Dtype> > add_diff(ParamSet<mode, Dtype>& pset, std::string param_name, 
												std::initializer_list<IdxType> l, Args&&... args)
{
	auto p = std::make_shared< ParamType<mode, Dtype> >(param_name, l, std::forward<Args>(args)...);
	pset.AddParam(p);
	return p;
}

}

#endif