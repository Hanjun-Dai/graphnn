#ifndef CPU_UNARY_FUNCTOR_H
#define CPU_UNARY_FUNCTOR_H

#include "unary_functor.h"
#include <chrono>
#include <cmath>

namespace gnn
{

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
	UnarySet(Dtype _scalar) : scalar(_scalar) {}

	/**
	 * set scalar to dst
	 */
	inline void operator()(Dtype& dst)
	{
		dst = scalar;
	}

private:
	/**
	 * scalar to be set
	 */
	Dtype scalar;
};

/**
 * @brief      CPU specialization of UnaryTruncate
 *
 * @tparam     Dtype  { float/double/int }
 */
template<typename Dtype>
class UnaryTruncate<CPU, Dtype>
{
public:
	UnaryTruncate(Dtype lb, Dtype ub) : lower_bound(lb), upper_bound(ub) {}

	/**
	 * truncate dst
	 */
	inline void operator()(Dtype& dst)
	{
		dst = dst < lower_bound ? lower_bound : dst;
		dst = dst > upper_bound ? upper_bound : dst;
	}

private:
	/**
	 * lower bound
	 */
	Dtype lower_bound;
	/**
	 * upper bound
	 */
	Dtype upper_bound;
};

/**
 * @brief      CPU specialization of UnarySigmoid
 *
 * @tparam     Dtype  { float/double/int }
 */
template<typename Dtype>
class UnarySigmoid<CPU, Dtype>
{
public:
	/**
	 * dst = 1 / (1 + exp(-dst))
	 */
	inline void operator()(Dtype& dst)
	{
		dst = 1.0 / (1.0 + exp(-dst));
	}
};

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
	UnaryRandNorm(Dtype _mean, Dtype _std) : mean(_mean), std(_std)
	{
		auto seed = std::chrono::system_clock::now().time_since_epoch().count();
		generator = new std::default_random_engine(seed);

		distribution = new std::normal_distribution<Dtype>(mean, std);
	}

	/**
	 * set set to be a sample from gaussian distribution
	 */
	inline void operator()(Dtype& dst)
	{
		auto& dist = *distribution;
		auto& engine = *generator;
		dst = dist(engine);
	}

private:
	std::default_random_engine* generator;
	std::normal_distribution<Dtype>* distribution;
	Dtype mean;
	Dtype std;
};

/**
 * @brief      CPU specialization of UnaryRandUniform
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnaryRandUniform<CPU, Dtype>
{
public:
	/**
	 * @brief      constructor
	 *
	 * @param[in]  _lb  The lower bound
	 * @param[in]  _ub   The upper bound
	 */
	UnaryRandUniform(Dtype _lb, Dtype _ub) : lb(_lb), ub(_ub) 
	{
		auto seed = std::chrono::system_clock::now().time_since_epoch().count();
		generator = new std::default_random_engine(seed);
		distribution = new std::uniform_real_distribution<Dtype>(lb, ub);
	}

	/**
	 * set set to be a sample from uniform distribution
	 */
	inline void operator()(Dtype& dst)
	{
		auto& dist = *distribution;
		auto& engine = *generator;
		dst = dist(engine);
	}

private:
	std::default_random_engine* generator;
	std::uniform_real_distribution<Dtype>* distribution;
	Dtype lb;
	Dtype ub;
};

//=================================== CPU call interface ======================================

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