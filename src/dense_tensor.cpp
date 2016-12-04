#include "dense_tensor.h"
#include "t_data.h"
#include "mem_holder.h"
#include <cstring>
#include <cassert>

namespace gnn 
{

template<typename Dtype>
TensorTemplate<CPU, DENSE, Dtype>::TensorTemplate()
{

}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Reshape(std::initializer_list<uint> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<CPU, Dtype> >();

	auto& t_data = this->data->Derived(this);
	if (this->shape.Count() > t_data.mem_size)
	{
		t_data.mem_size = this->shape.Count();
		MemHolder<CPU>::DelArr(t_data.ptr);
		MemHolder<CPU>::MallocArr(t_data.ptr, sizeof(Dtype) * this->shape.Count());
	}
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Zeros()
{
	auto& t_data = this->data->Derived(this);

	if (t_data.mem_size)
	   memset(t_data.ptr, 0, sizeof(Dtype) * t_data.mem_size);
}

template<typename Dtype>
int TensorTemplate<CPU, DENSE, Dtype>::AsInt()
{
	assert(this->shape.Count() == 1);
	auto& t_data = this->data->Derived(this);
	return t_data.ptr[0];	
}

template class TensorTemplate<CPU, DENSE, float>;
template class TensorTemplate<CPU, DENSE, double>;
template class TensorTemplate<CPU, DENSE, int>;

}