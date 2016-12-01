#include "dense_tensor.h"
#include "t_data.h"

namespace gnn 
{

template<DataType dType>
TensorTemplate<CPU, DENSE, dType>::TensorTemplate()
{

}

template<DataType dType>
void TensorTemplate<CPU, DENSE, dType>::Reshape(std::initializer_list<uint> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<CPU, FLOAT32> >();

}

template<DataType dType>
void TensorTemplate<CPU, DENSE, dType>::Zeros()
{
	std::cerr << "zeros" << std::endl;
}

template class TensorTemplate<CPU, DENSE, FLOAT32>;
template class TensorTemplate<CPU, DENSE, FLOAT64>;
template class TensorTemplate<CPU, DENSE, INT32>;

}