#ifndef DENSE_TENSOR_H
#define DENSE_TENSOR_H

#include "tensor.h"

namespace gnn
{

template<DataType dType>
class TensorTemplate<CPU, DENSE, dType> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::initializer_list<uint> l) override;

	virtual void Zeros() override;
};

}
#endif