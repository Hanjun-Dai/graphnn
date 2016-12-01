#include "tensor.h"
#include "t_data.h"
#include "dense_tensor.h"

namespace gnn
{

template<Mode mode, MatType matType, DataType dType>
TensorTemplate<mode, matType, dType>& Tensor::Derived()
{
	return *(dynamic_cast<TensorTemplate<mode, matType, dType>*>(this));
}

template TensorTemplate<CPU, DENSE, FLOAT32>& Tensor::Derived<CPU, DENSE, FLOAT32>(); 
template TensorTemplate<CPU, DENSE, FLOAT64>& Tensor::Derived<CPU, DENSE, FLOAT64>(); 
template TensorTemplate<CPU, DENSE, INT32>& Tensor::Derived<CPU, DENSE, INT32>(); 

}