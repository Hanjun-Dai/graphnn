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

template<typename mode, typename matType, typename Dtype>
class TensorTemplate;

template<typename mode, typename Dtype>
using DenseTensor = TensorTemplate<mode, DENSE, Dtype>;

template<typename mode, typename Dtype>
using SparseTensor = TensorTemplate<mode, SPARSE, Dtype>;

/**
 * @brief      Class for tensor.
 */
class Tensor
{
public:
// 
	template<typename mode, typename matType, typename Dtype>
	TensorTemplate<mode, matType, Dtype>& Derived();

	virtual void Reshape(std::initializer_list<uint> l) NOT_IMPLEMENTED

	virtual void Zeros() NOT_IMPLEMENTED

	virtual int AsInt() NOT_IMPLEMENTED

	template<typename Dtype>
	Dtype AsScalar(); 

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

template<typename mode, typename matType, typename Dtype>
class TensorTemplate : public Tensor {};


} // namespace gnn
#endif