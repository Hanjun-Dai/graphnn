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

template<typename mode, typename matType>
class TensorTemplate;

template<typename mode>
using DenseTensor = TensorTemplate<mode, DENSE>;

template<typename mode>
using SparseTensor = TensorTemplate<mode, SPARSE>;

/**
 * @brief      Abstract Class for tensor.
 */
class Tensor
{
public:
// 
	template<typename mode, typename matType>
	TensorTemplate<mode, matType>& Derived();

	virtual void Reshape(std::initializer_list<uint>) NOT_IMPLEMENTED

	virtual void Zeros() NOT_IMPLEMENTED

	virtual Dtype AsScalar() NOT_IMPLEMENTED

	virtual void SetRandN(Dtype, Dtype) NOT_IMPLEMENTED

	virtual void Fill(Dtype) NOT_IMPLEMENTED

	/**
	 * Tensor shape
	 */
	TShape shape;
	/**
	 * Tensor rank
	 */
	uint rank;
	/**
	 * data ptr
	 */
	std::shared_ptr< TData > data;
private:

};

template<typename mode, typename matType>
class TensorTemplate : public Tensor {};


} // namespace gnn
#endif