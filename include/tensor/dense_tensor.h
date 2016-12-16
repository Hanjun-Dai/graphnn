#ifndef DENSE_TENSOR_H
#define DENSE_TENSOR_H

#include "tensor.h"
#include "t_data.h"

namespace gnn
{


template<typename Dtype>
class TensorTemplate<CPU, DENSE, Dtype> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::initializer_list<uint> l) override;

	virtual void Zeros(); 

	virtual Dtype AsScalar(); 

	virtual void SetRandN(Dtype mean, Dtype std);

	virtual void Fill(Dtype scalar);

	virtual Dtype ASum();

	std::shared_ptr< DenseData<CPU, Dtype> > data;
};

template<>
class TensorTemplate<CPU, DENSE, int> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::initializer_list<uint> l) override;

	virtual void Zeros(); 

	virtual int AsScalar(); 

	virtual void Fill(int scalar);

	std::shared_ptr< DenseData<CPU, int> > data;
};

}
#endif