#ifndef PARAM_SET_H
#define PARAM_SET_H

#include <string>
#include <map>
#include "nn/variable.h"

namespace gnn
{

/**
 * @brief      the set of learnable params
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class ParamSet
{
public:
	ParamSet();

	/**
	 * @brief      Adds a parameter.
	 *
	 * @param[in]  param  The parameter shared pointer
	 */
	void AddParam(std::shared_ptr< DTensorVar<mode, Dtype> > param);

	/**
	 * @brief      Adds a non differentiable param
	 *
	 * @param[in]  param  The parameter
	 */
	void AddNondiff(std::shared_ptr< DTensorVar<mode, Dtype> > param);

	/**
	 * @brief      save the params into disk
	 *
	 * @param[in]  filename  The filename
	 */
	void Save(std::string filename);

	/**
	 * @brief      load params from disk
	 *
	 * @param[in]  filename  The filename
	 */
	void Load(std::string filename);

	/**
	 * @brief      deeply copy from another param set
	 *
	 * @param      src   The source param set
	 */
	void DeepCopyFrom(ParamSet<mode, Dtype>& src);
	
	/**
	 * the dictionary: param name -> param shared pointer
	 */
	std::map<std::string, std::shared_ptr< DTensorVar<mode, Dtype> > > params;

	/**
	 * the dictionary: nondiff param name -> nondiff param shared pointer
	 */
	std::map<std::string, std::shared_ptr< DTensorVar<mode, Dtype> > > nondiff_params;
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

template<template <typename, typename> class ParamType, typename mode, typename Dtype, typename... Args>
std::shared_ptr< ParamType<mode, Dtype> > add_nondiff(ParamSet<mode, Dtype>& pset, std::string param_name, Args&&... args)
{
	auto p = std::make_shared< ParamType<mode, Dtype> >(param_name, std::forward<Args>(args)...);
	pset.AddNondiff(p);
	return p;
}

template<template <typename, typename> class ParamType, typename mode, typename Dtype, typename IdxType, typename... Args>
std::shared_ptr< ParamType<mode, Dtype> > add_nondiff(ParamSet<mode, Dtype>& pset, std::string param_name, 
												std::initializer_list<IdxType> l, Args&&... args)
{
	auto p = std::make_shared< ParamType<mode, Dtype> >(param_name, l, std::forward<Args>(args)...);
	pset.AddNondiff(p);
	return p;
}

}

#endif