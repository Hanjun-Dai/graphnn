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

	virtual void Reshape(std::initializer_list<uint> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;
	
	void CopyFrom(SpTensor<CPU, Dtype>& src);
	
	void ResizeSp(int newNNZ, int newNPtr);
	std::shared_ptr< SparseData<CPU, Dtype> > data;
};

}

#endif