#include "tensor.h"
#include "t_data.h"
#include "dense_tensor.h"

namespace gnn
{

Tensor::Tensor()
{

}

template<Mode mode, MatType matType, uint rank, DataType dType>
TensorTemplate<mode, matType, rank, dType> Tensor::Get()
{
	TensorTemplate<mode, matType, rank, dType> result(shape, data);
	return result;
}


Tensor operator+(const Tensor& lhs, const Tensor& rhs)
{
	Tensor result;
	return result;
}

template TensorTemplate<CPU, DENSE, 1, FLOAT32> Tensor::Get<CPU, DENSE, 1, FLOAT32>(); 

}