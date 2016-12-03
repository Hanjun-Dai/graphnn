#ifndef DENSE_TENSOR_H
#define DENSE_TENSOR_H

#include "tensor.h"

namespace gnn
{

template<typename Dtype>
class TensorTemplate<CPU, DENSE, Dtype> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::initializer_list<uint> l) override;

	virtual void Zeros() override;

	virtual int AsInt() override;

};

}
#endif