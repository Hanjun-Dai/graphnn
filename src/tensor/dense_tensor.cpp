#include "tensor/dense_tensor.h"
#include "tensor/t_data.h"
#include "tensor/unary_functor.h"
#include "tensor/mkl_helper.h"
#include "util/mem_holder.h"
#include <cstring>
#include <cassert>

namespace gnn 
{

template<typename Dtype>
TensorTemplate<CPU, DENSE, Dtype>::TensorTemplate() : data(nullptr)
{
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Reshape(std::initializer_list<uint> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<CPU, Dtype> >();

	if (this->shape.Count() > this->data->mem_size)
	{
		this->data->mem_size = this->shape.Count();
		MemHolder<CPU>::DelArr(this->data->ptr);
		MemHolder<CPU>::MallocArr(this->data->ptr, sizeof(Dtype) * this->shape.Count());
	}
}

template<typename Dtype>
MatType TensorTemplate<CPU, DENSE, Dtype>::GetMatType()
{
	return MatType::dense;
}

template<typename Dtype>
MatMode TensorTemplate<CPU, DENSE, Dtype>::GetMatMode()
{
	return MatMode::cpu;
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Zeros()
{
	if (this->data->mem_size)
	   memset(this->data->ptr, 0, sizeof(Dtype) * this->data->mem_size);
}

template<typename Dtype>
Dtype TensorTemplate<CPU, DENSE, Dtype>::AsScalar()
{
	assert(this->shape.Count() == 1);
	return this->data->ptr[0];	
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::SetRandN(Dtype mean, Dtype std)
{

}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Fill(Dtype scalar)
{
	if (scalar == 0)
		this->Zeros();
	else {
		UnaryEngine<CPU>::Exec<UnarySet>(this->data->ptr, this->shape.Count(), scalar);
	}
}

template<typename Dtype>
Dtype TensorTemplate<CPU, DENSE, Dtype>::ASum()
{
	return MKL_ASum(this->shape.Count(), this->data->ptr);
}

template class TensorTemplate<CPU, DENSE, float>;
template class TensorTemplate<CPU, DENSE, double>;

///================================ int tensor ===================================

TensorTemplate<CPU, DENSE, int>::TensorTemplate() : data(nullptr)
{

}

void TensorTemplate<CPU, DENSE, int>::Reshape(std::initializer_list<uint> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<CPU, int> >();

	if (this->shape.Count() > this->data->mem_size)
	{
		this->data->mem_size = this->shape.Count();
		MemHolder<CPU>::DelArr(this->data->ptr);
		MemHolder<CPU>::MallocArr(this->data->ptr, sizeof(int) * this->shape.Count());
	}
}

MatType TensorTemplate<CPU, DENSE, int>::GetMatType()
{
	return MatType::dense;
}

MatMode TensorTemplate<CPU, DENSE, int>::GetMatMode()
{
	return MatMode::cpu;
}

void TensorTemplate<CPU, DENSE, int>::Zeros()
{
	if (this->data->mem_size)
	   memset(this->data->ptr, 0, sizeof(int) * this->data->mem_size);
}

void TensorTemplate<CPU, DENSE, int>::Fill(int scalar)
{
	if (scalar == 0)
		this->Zeros();
	else {
		UnaryEngine<CPU>::Exec<UnarySet>(this->data->ptr, this->shape.Count(), scalar);
	}
}

int TensorTemplate<CPU, DENSE, int>::AsScalar()
{
	assert(this->shape.Count() == 1);
	return this->data->ptr[0];	
}

template class TensorTemplate<CPU, DENSE, int>;

}