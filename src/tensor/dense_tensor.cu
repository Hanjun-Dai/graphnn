#include "tensor/dense_tensor.h"
#include "tensor/t_data.h"
#include "util/mem_holder.h"

namespace gnn
{
	
template<typename Dtype>
TensorTemplate<GPU, DENSE, Dtype>::TensorTemplate() : data(nullptr)
{
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<GPU, Dtype> >();

	this->data->Resize(this->shape.Count());
}

template<typename Dtype>
MatType TensorTemplate<GPU, DENSE, Dtype>::GetMatType()
{
	return MatType::dense;
}

template<typename Dtype>
MatMode TensorTemplate<GPU, DENSE, Dtype>::GetMatMode()
{
	return MatMode::gpu;
}

template class TensorTemplate<GPU, DENSE, float>;
template class TensorTemplate<GPU, DENSE, double>;

///================================ int tensor ===================================

TensorTemplate<GPU, DENSE, int>::TensorTemplate() : data(nullptr)
{

}

void TensorTemplate<GPU, DENSE, int>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<GPU, int> >();

	if (this->shape.Count() > this->data->mem_size)
	{
		this->data->mem_size = this->shape.Count();
		MemHolder<GPU>::DelArr(this->data->ptr);
		MemHolder<GPU>::MallocArr(this->data->ptr, sizeof(int) * this->shape.Count());
	}
}

MatType TensorTemplate<GPU, DENSE, int>::GetMatType()
{
	return MatType::dense;
}

MatMode TensorTemplate<GPU, DENSE, int>::GetMatMode()
{
	return MatMode::gpu;
}

template class TensorTemplate<GPU, DENSE, int>;

}