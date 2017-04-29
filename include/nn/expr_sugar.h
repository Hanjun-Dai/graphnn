#ifndef EXPR_SUGAR_H
#define EXPR_SUGAR_H

#include "nn/factor_graph.h"
#include "nn/type_cast.h"

namespace gnn
{

template<template <typename, typename> class FacType, typename... Args>
typename FacType<CPU, float>::OutType af(FactorGraph& g, std::vector< std::shared_ptr< TensorVar<CPU, float> > > op, 
										Args&&... args)
{
	return af< FacType<CPU, float> >(g, op, std::forward<Args>(args)...);
}

template<template <typename, typename> class FacType, typename... Args>
typename FacType<CPU, double>::OutType af(FactorGraph& g, std::vector< std::shared_ptr< TensorVar<CPU, double> > > op, 
										Args&&... args)
{
	return af< FacType<CPU, double> >(g, op, std::forward<Args>(args)...);
}

template<template <typename, typename> class FacType, typename... Args>
typename FacType<CPU, int>::OutType af(FactorGraph& g, std::vector< std::shared_ptr< TensorVar<CPU, int> > > op, 
										Args&&... args)
{
	return af< FacType<CPU, int> >(g, op, std::forward<Args>(args)...);
}

#ifdef USE_GPU
template<template <typename, typename> class FacType, typename... Args>
typename FacType<GPU, float>::OutType af(FactorGraph& g, std::vector< std::shared_ptr< TensorVar<GPU, float> > > op, 
										Args&&... args)
{
	return af< FacType<GPU, float> >(g, op, std::forward<Args>(args)...);
}

template<template <typename, typename> class FacType, typename... Args>
typename FacType<GPU, double>::OutType af(FactorGraph& g, std::vector< std::shared_ptr< TensorVar<GPU, double> > > op, 
										Args&&... args)
{
	return af< FacType<GPU, double> >(g, op, std::forward<Args>(args)...);
}

template<template <typename, typename> class FacType, typename... Args>
typename FacType<GPU, int>::OutType af(FactorGraph& g, std::vector< std::shared_ptr< TensorVar<GPU, int> > > op, 
										Args&&... args)
{
	return af< FacType<GPU, int> >(g, op, std::forward<Args>(args)...);
}
#endif

}

#endif