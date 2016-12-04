#include "dense_tensor.h"
#include "t_data.h"
#include "mem_holder.h"
#include <cstring>
#include <cassert>

namespace gnn 
{

TensorTemplate<CPU, DENSE>::TensorTemplate()
{

}

void TensorTemplate<CPU, DENSE>::Reshape(std::initializer_list<uint> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<CPU> >();

	auto& t_data = this->data->Derived(this);
	if (this->shape.Count() > t_data.mem_size)
	{
		t_data.mem_size = this->shape.Count();
		MemHolder<CPU>::DelArr(t_data.ptr);
		MemHolder<CPU>::MallocArr(t_data.ptr, sizeof(Dtype) * this->shape.Count());
	}
}

void TensorTemplate<CPU, DENSE>::Zeros()
{
	auto& t_data = this->data->Derived(this);

	if (t_data.mem_size)
	   memset(t_data.ptr, 0, sizeof(Dtype) * t_data.mem_size);
}

Dtype TensorTemplate<CPU, DENSE>::AsScalar()
{
	assert(this->shape.Count() == 1);
	auto& t_data = this->data->Derived(this);
	return t_data.ptr[0];	
}

void TensorTemplate<CPU, DENSE>::SetRandN(Dtype mean, Dtype std)
{

}

void TensorTemplate<CPU, DENSE>::Fill(Dtype scalar)
{
	
}

template class TensorTemplate<CPU, DENSE>;

}