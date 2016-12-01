#ifndef TENSOR_H
#define TENSOR_H

#include "t_shape.h"
#include <memory>
#include <iostream>

namespace gnn {

typedef unsigned int uint;

enum Mode
{
	CPU = 0,
	GPU = 1
};

enum MatType
{
	DENSE,
	SPARSE
};

enum DataType
{
	FLOAT32,
	FLOAT64,
	INT32
};

class TData;

template<Mode mode, MatType matType, uint rank, DataType dType>
class TensorTemplate;

template<Mode mode, uint rank, DataType dType>
using DenseTensor = TensorTemplate<mode, DENSE, rank, dType>;

/**
 * @brief      Class for tensor.
 */
class Tensor
{
public:

	Tensor();

	virtual void Print() { std::cerr << "base_class" << std::endl; };

	template<Mode mode, MatType matType, uint rank, DataType dType>
	TensorTemplate<mode, matType, rank, dType> Get();
	
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

template<Mode mode, MatType matType, uint rank, DataType dType>
class TensorTemplate : public Tensor {};


Tensor operator+(const Tensor& lhs, const Tensor& rhs);

} // namespace gnn
#endif