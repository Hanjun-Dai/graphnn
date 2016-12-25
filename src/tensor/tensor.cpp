#include "tensor/tensor.h"
#include "tensor/t_data.h"
#include "tensor/cpu_dense_tensor.h"

namespace gnn
{

template<typename mode, typename matType, typename Dtype>
TensorTemplate<mode, matType, Dtype>& Tensor::Derived()
{
	return *(dynamic_cast<TensorTemplate<mode, matType, Dtype>*>(this));
}

template TensorTemplate<CPU, DENSE, float>& Tensor::Derived<CPU, DENSE, float>(); 
template TensorTemplate<CPU, DENSE, double>& Tensor::Derived<CPU, DENSE, double>(); 
template TensorTemplate<CPU, DENSE, int>& Tensor::Derived<CPU, DENSE, int>(); 
template TensorTemplate<GPU, DENSE, float>& Tensor::Derived<GPU, DENSE, float>(); 
template TensorTemplate<GPU, DENSE, double>& Tensor::Derived<GPU, DENSE, double>(); 
template TensorTemplate<GPU, DENSE, int>& Tensor::Derived<GPU, DENSE, int>(); 
}