#ifndef DENSE_TENSOR_H
#define DENSE_TENSOR_H

#include "tensor.h"

namespace gnn
{

template<>
class TensorTemplate<CPU, DENSE> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::initializer_list<uint> l) override;

	virtual void Zeros() override;

	virtual Dtype AsScalar() override;

	virtual void SetRandN(Dtype mean, Dtype std) override;

	virtual void Fill(Dtype scalar) override;
};

}
#endif