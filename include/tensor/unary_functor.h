#ifndef UNARY_FUNCTOR_H
#define UNARY_FUNCTOR_H

#include "util/gnn_macros.h"
#include <utility>
#include <random>

namespace gnn
{

/**
 * @brief      Functor to set an element
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double/int }
 */
template<typename mode, typename Dtype>
class UnarySet
{};

/**
 * @brief      CPU specialization of UnarySet
 *
 * @tparam     Dtype  { float/double/int }
 */
template<typename Dtype>
class UnarySet<CPU, Dtype>
{
public:
	/**
	 * @brief      constructor
	 *
	 * @param[in]  _scalar  The scalar to be set
	 */
	UnarySet(Dtype _scalar);

	/**
	 * set scalar to dst
	 */
	void operator()(Dtype& dst);

private:
	/**
	 * scalar to be set
	 */
	Dtype scalar;
};

/**
 * @brief      Functor to set an element from normal distribution
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryRandNorm {};

/**
 * @brief      CPU specialization of UnaryRandNorm
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnaryRandNorm<CPU, Dtype>
{
public:
	/**
	 * @brief      constructor
	 *
	 * @param[in]  _mean  The mean
	 * @param[in]  _std   The standard deviation
	 */
	UnaryRandNorm(Dtype _mean, Dtype _std);

	/**
	 * set set to be a sample from gaussian distribution
	 */
	void operator()(Dtype& dst);

private:
	std::default_random_engine* generator;
	std::normal_distribution<Dtype>* distribution;
	Dtype mean;
	Dtype std;
};

/**
 * @brief      Class for unary engine.
 *
 * @tparam     mode  CPU/GPU
 */
template<typename mode>
class UnaryEngine
{};

/**
 * @brief      Class for unary engine, CPU specialization
 */
template<>
class UnaryEngine<CPU>
{
public:
	/**
	 * @brief      Execute the unary operation
	 *
	 * @param      data       The data pointer
	 * @param[in]  count      # elements to operate
	 * @param[in]  <unnamed>  { other parameter required by specific functor }
	 *
	 * @tparam     Functor    { Functor class type }
	 * @tparam     Dtype      { float/double/int }
	 * @tparam     Args       { extra arguements required by specific functor }
	 */
	template<template <typename, typename> class Functor, typename Dtype, typename... Args>
	static void Exec(Dtype* data, size_t count, Args&&... args)
	{
		Functor<CPU, Dtype> func(std::forward<Args>(args)...);
		Exec(data, count, func);
	}

	/**
	 * @brief      Execute the unary operation
	 *
	 * @param      data     The data pointer 
	 * @param[in]  count    # elements to operate
	 * @param[in]  f        { the functor object }
	 *
	 * @tparam     Dtype    { float/double/int }
	 * @tparam     Functor  { The functor class }
	 */
	template<typename Dtype, typename Functor>
	static void Exec(Dtype* data, size_t count, Functor f)
	{
		for (size_t i = 0; i < count; ++i)
			f(data[i]);
	}
};

}

#endif