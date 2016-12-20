#ifndef UNARY_FUNCTOR_H
#define UNARY_FUNCTOR_H

#include "util/gnn_macros.h"
#include <utility>
#include <random>

namespace gnn
{


template<typename mode, typename Dtype>
class UnarySet
{};


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

template<typename mode, typename Dtype>
class UnaryRandNorm {};

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
	template<template <typename, typename> class Functor, typename Dtype, typename... Args>
	static void Exec(Dtype* data, size_t count, Args&&... args)
	{
		Functor<CPU, Dtype> func(std::forward<Args>(args)...);
		Exec(data, count, func);
	}

	template<typename Dtype, typename Functor>
	static void Exec(Dtype* data, size_t count, Functor f)
	{
		for (size_t i = 0; i < count; ++i)
			f(data[i]);
	}
};

}

#endif