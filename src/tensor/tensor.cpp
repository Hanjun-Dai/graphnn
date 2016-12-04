#include "tensor/tensor.h"
#include "tensor/t_data.h"
#include "tensor/dense_tensor.h"

namespace gnn
{

template<typename mode, typename matType>
TensorTemplate<mode, matType>& Tensor::Derived()
{
	return *(dynamic_cast<TensorTemplate<mode, matType>*>(this));
}

template TensorTemplate<CPU, DENSE>& Tensor::Derived<CPU, DENSE>(); 
}