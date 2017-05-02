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
class UnarySet {};

/**
 * @brief      Functor to scale an element
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double/int }
 */
template<typename mode, typename Dtype>
class UnaryScale {};

/**
 * @brief      Functor to add an element
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double/int }
 */
template<typename mode, typename Dtype>
class UnaryAdd {};

/**
 * @brief      Functor to set an element from normal distribution
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryRandNorm {};

/**
 * @brief      Functor to set an element from uniform distribution
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryRandUniform {};

/**
 * @brief      Functor to abs an element; CPU doesn't need this;
 *
 * @tparam     mode   { GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryAbs {};

/**
 * @brief      Functor to invert an element; CPU doesn't need this;
 *
 * @tparam     mode   { GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryInv {};

/**
 * @brief      relu
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryReLU {};

/**
 * @brief      sigmoid
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnarySigmoid {};

/**
 * @brief      tanh
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryTanh {};

/**
 * @brief      Functor to square an element; CPU doesn't need this;
 *
 * @tparam     mode   { GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnarySquare {};

/**
 * @brief      Functor to sqrt an element; CPU doesn't need this;
 *
 * @tparam     mode   { GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnarySqrt {};

/**
 * @brief      Functor to inv_sqrt an element; CPU doesn't need this;
 *
 * @tparam     mode   { GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryInvSqrt {};


/**
 * @brief      Functor to log an element; CPU doesn't need this;
 *
 * @tparam     mode   { GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryLog {};

/**
 * @brief      Functor to exp an element; CPU doesn't need this;
 *
 * @tparam     mode   { GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryExp {};

/**
 * @brief      Functor to truncate an element;
 *
 * @tparam     mode   { GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryTruncate {};

/**
 * @brief      Class for unary engine.
 *
 * @tparam     mode  CPU/GPU
 */
template<typename mode>
class UnaryEngine
{};

}

#endif