#ifndef TENSOR_H
#define TENSOR_H

#include "util/gnn_macros.h"
#include "t_shape.h"
#include <memory>
#include <iostream>
#include <exception>
#include <string>

namespace gnn {
	
template<typename mode, typename matType, typename Dtype>
class TensorTemplate;

template<typename mode, typename Dtype>
using DTensor = TensorTemplate<mode, DENSE, Dtype>;

template<typename mode, typename Dtype>
using SpTensor = TensorTemplate<mode, SPARSE, Dtype>;

/**
 * @brief      Abstract Class for tensor.
 */
class Tensor
{
public:
	template<typename mode, typename matType, typename Dtype>
	TensorTemplate<mode, matType, Dtype>& Derived();

	virtual void Reshape(std::initializer_list<uint> l) NOT_IMPLEMENTED

	virtual MatType GetMatType() = 0;

	virtual MatMode GetMatMode() = 0;

	/**
	 * Tensor shape
	 */
	TShape shape;
private:

};

template<typename mode, typename matType, typename Dtype>
class TensorTemplate : public Tensor {};


} // namespace gnn
#endif