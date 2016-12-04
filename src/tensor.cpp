#include "tensor.h"
#include "t_data.h"
#include "dense_tensor.h"

namespace gnn
{

template<typename mode, typename matType>
TensorTemplate<mode, matType>& Tensor::Derived()
{
	return *(dynamic_cast<TensorTemplate<mode, matType>*>(this));
}

template TensorTemplate<CPU, DENSE>& Tensor::Derived<CPU, DENSE>(); 
}