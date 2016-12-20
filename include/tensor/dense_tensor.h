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

	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;

	void CopyFrom(DTensor<CPU, Dtype>& src);		
	void CopyFrom(DTensor<GPU, Dtype>& src);

	template<typename otherType> 
	void CopyFrom(DTensor<CPU, otherType>& src)
	{
		Reshape(src.shape.dims);
		for (size_t i = 0; i < src.shape.Count(); ++i)
			data->ptr[i] = src.data->ptr[i];
	}

	void ShallowCopy(DTensor<CPU, Dtype>& src);

	void Zeros(); 

	Dtype AsScalar(); 

	void SetRandN(Dtype mean, Dtype std);

	void Fill(Dtype scalar);

	Dtype ASum();

	void ArgMax(DTensor<CPU, int>& dst, uint axis = 0);

	void MM(DTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta);
	void MM(SpTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta);

	void Softmax();
	
	void Mean(DTensor<CPU, Dtype>& a, int axis = -1);

	std::shared_ptr< DenseData<CPU, Dtype> > data;
};

template<>
class TensorTemplate<CPU, DENSE, int> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;

	void ShallowCopy(DTensor<CPU, int>& src);

	void Zeros(); 

	int AsScalar(); 

	void Fill(int scalar);

	std::shared_ptr< DenseData<CPU, int> > data;
};

template<typename Dtype>
class TensorTemplate<GPU, DENSE, Dtype> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;

	std::shared_ptr< DenseData<GPU, Dtype> > data;
};

template<>
class TensorTemplate<GPU, DENSE, int> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;
	
	std::shared_ptr< DenseData<GPU, int> > data;
};

}
#endif