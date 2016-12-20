#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include "tensor.h"
#include "t_data.h"

namespace gnn
{

template<typename Dtype>
class TensorTemplate<CPU, SPARSE, Dtype> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;
	
	void CopyFrom(SpTensor<CPU, Dtype>& src);
	void ShallowCopy(SpTensor<CPU, Dtype>& src);
	
	void ResizeSp(int newNNZ, int newNPtr);

	void ArgMax(DTensor<CPU, int>& dst, uint axis = 0);

	std::shared_ptr< SparseData<CPU, Dtype> > data;
};

template<>
class TensorTemplate<CPU, SPARSE, int> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;
	
	void ShallowCopy(SpTensor<CPU, int>& src);
	
	void ResizeSp(int newNNZ, int newNPtr);
	std::shared_ptr< SparseData<CPU, int> > data;
};

}

#endif