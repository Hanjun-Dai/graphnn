#include "tensor.h"
#include "t_data.h"
#include "dense_tensor.h"

namespace gnn
{

template<Mode mode, MatType matType, typename Dtype>
TensorTemplate<mode, matType, Dtype>& Tensor::Derived()
{
	return *(dynamic_cast<TensorTemplate<mode, matType, Dtype>*>(this));
}

template TensorTemplate<CPU, DENSE, float>& Tensor::Derived<CPU, DENSE, float>(); 
template TensorTemplate<CPU, DENSE, double>& Tensor::Derived<CPU, DENSE, double>(); 
template TensorTemplate<CPU, DENSE, int>& Tensor::Derived<CPU, DENSE, int>(); 

}