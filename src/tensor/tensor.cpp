#include "tensor/tensor.h"
#include "tensor/t_data.h"
#include "tensor/dense_tensor.h"

namespace gnn
{

template<typename mode, typename matType, typename Dtype>
TensorTemplate<mode, matType, Dtype>& Tensor::Derived()
{
	return *(dynamic_cast<TensorTemplate<mode, matType, Dtype>*>(this));
}

uint Tensor::rank()
{
	return this->shape.dims.size();
}

size_t Tensor::rows()
{
	ASSERT(rank() <= 2, "rows is not well defined for rank > 2");
	return this->shape[0];
}

size_t Tensor::cols()
{
	// assume column vector
	if (rank() == 1)
		return 1;
	if (rank() == 2)
		return this->shape[1];
	ASSERT(false, "cols is not well defined for rank > 2");
}


template TensorTemplate<CPU, DENSE, float>& Tensor::Derived<CPU, DENSE, float>(); 
template TensorTemplate<CPU, DENSE, double>& Tensor::Derived<CPU, DENSE, double>(); 
template TensorTemplate<CPU, DENSE, int>& Tensor::Derived<CPU, DENSE, int>(); 

}