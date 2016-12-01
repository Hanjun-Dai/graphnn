#ifndef TENSOR_H
#define TENSOR_H

#include "gnn_macros.h"
#include "t_shape.h"
#include <memory>
#include <iostream>
#include <exception>
#include <string>

namespace gnn {

class TData;

template<Mode mode, MatType matType, typename Dtype>
class TensorTemplate;

template<Mode mode, typename Dtype>
using DenseTensor = TensorTemplate<mode, DENSE, Dtype>;

template<Mode mode, typename Dtype>
using SparseTensor = TensorTemplate<mode, SPARSE, Dtype>;

/**
 * @brief      Class for tensor.
 */
class Tensor
{
public:

	template<Mode mode, MatType matType, typename Dtype>
	TensorTemplate<mode, matType, Dtype>& Derived();

	virtual void Reshape(std::initializer_list<uint> l) NOT_IMPLEMENTED

	virtual void Zeros() NOT_IMPLEMENTED

	/**
	 * Tensor shape
	 */
	TShape shape;
	/**
	 * Tensor rank
	 */
	uint rank;
	/**
	 * cpu or gpu mode
	 */
	Mode mode;
	/**
	 * sparse or dense
	 */
	MatType matType;
	/**
	 * data ptr
	 */
	std::shared_ptr< TData > data;
private:

};

template<Mode mode, MatType matType, typename Dtype>
class TensorTemplate : public Tensor {};


} // namespace gnn
#endif