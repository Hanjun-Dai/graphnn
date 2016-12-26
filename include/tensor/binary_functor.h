#ifndef BINARY_FUNCTOR_H
#define BINARY_FUNCTOR_H

#include "util/gnn_macros.h"
#include <utility>

namespace gnn
{

/**
 * @brief      Class for binary mul.
 *
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class BinaryMul
{};

/**
 * @brief      Class for binary engine.
 *
 * @tparam     mode  CPU/GPU
 */
template<typename mode>
class BinaryEngine
{};

}

#endif