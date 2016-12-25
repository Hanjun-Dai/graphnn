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
 * @brief      Functor to invert an element; CPU doesn't need this;
 *
 * @tparam     mode   { GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class UnaryInv {};

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
 * @brief      Class for unary engine.
 *
 * @tparam     mode  CPU/GPU
 */
template<typename mode>
class UnaryEngine
{};

}

#endif