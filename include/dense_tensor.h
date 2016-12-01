#ifndef DENSE_TENSOR_H
#define DENSE_TENSOR_H

#include "tensor.h"

namespace gnn
{

template<uint rank, DataType dType>
class TensorTemplate<CPU, DENSE, rank, dType> : public Tensor
{
public:

	virtual void Print() { std::cerr << "derived class" << std::endl; };

	TensorTemplate(TShape _shape, std::shared_ptr< TData > _data)
	{

	}
};

}
#endif