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

template<Mode mode, MatType matType, DataType dType>
class TensorTemplate;

template<Mode mode, DataType dType>
using DenseTensor = TensorTemplate<mode, DENSE, dType>;

template<Mode mode, DataType dType>
using SparseTensor = TensorTemplate<mode, SPARSE, dType>;

/**
 * @brief      Class for tensor.
 */
class Tensor
{
public:

	template<Mode mode, MatType matType, DataType dType>
	TensorTemplate<mode, matType, dType>& Derived();

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
	/**
	 * data type
	 */
	DataType dType;
private:

};

template<Mode mode, MatType matType, DataType dType>
class TensorTemplate : public Tensor {};


} // namespace gnn
#endif