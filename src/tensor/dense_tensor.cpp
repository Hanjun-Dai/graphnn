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
void TensorTemplate<CPU, DENSE, Dtype>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<CPU, Dtype> >();

	this->data->Resize(this->shape.Count());
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
void TensorTemplate<CPU, DENSE, Dtype>::CopyFrom(DTensor<CPU, Dtype>& src)
{
	Reshape(src.shape.dims);
	memcpy(data->ptr, src.data->ptr, sizeof(Dtype) * shape.Count());
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::CopyFrom(DTensor<GPU, Dtype>& src)
{
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::ShallowCopy(DTensor<CPU, Dtype>& src)
{
	this->shape = src.shape;
	this->data = src.data;
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
	UnaryEngine<CPU>::Exec<UnaryRandNorm>(this->data->ptr, this->shape.Count(), mean, std);
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

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::ArgMax(DTensor<CPU, int>& dst, uint axis)
{
	ASSERT(axis == 0, "not supported for axis > 0 in CPU DENSE Tensor");
	dst.Reshape({this->shape[0]});

	Dtype* ptr = data->ptr;
	for (size_t i = 0; i < this->shape[0]; ++i)
	{
		dst.data->ptr[i] = 0;
		Dtype cur_max = *ptr;
		for (size_t j = 1; j < this->shape.Count(1); ++j)
			if (ptr[j] > cur_max)
			{
				cur_max = ptr[j];
				dst.data->ptr[i] = j;
			}
		ptr += this->shape.Count(1);
	}
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::MM(DTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{
	ASSERT(a.rank() == 2 && b.rank() == 2, "only support mat x mat now");
	size_t m, n, k;
	GetDims(a.rows(), a.cols(), transA, b.rows(), b.cols(), transB, m, n, k);
	
	Reshape({m, n});
	MKL_GeMM(CblasRowMajor, CPU_T(transB), CPU_T(transB), 
			m, n, k, alpha, 
			a.data->ptr, a.cols(), 
			b.data->ptr, b.cols(), 
			beta, data->ptr, this->cols());
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::MM(SpTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{

}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Softmax()
{
	ASSERT(this->rank() == 2, "Softmax is assumed to exec on matrix");
    Dtype sum, max_v;
	size_t i, j;
	Dtype* data_ptr;
    for (i = 0, data_ptr = this->data->ptr; i < this->rows(); ++i, data_ptr += this->cols())
    {
        max_v = data_ptr[0];
        for (j = 1; j < this->cols(); ++j)
            if (data_ptr[j] > max_v)
                max_v = data_ptr[j];
        for (j = 0; j < this->cols(); ++j)    
            data_ptr[j] -= max_v;
    }
    
    MKL_Exp(this->shape.Count(), this->data->ptr, this->data->ptr);
    for (i = 0, data_ptr = this->data->ptr; i < this->rows(); ++i, data_ptr += this->cols())
    {
        sum = MKL_ASum(this->cols(), data_ptr);
        for (j = 0; j < this->cols(); ++j)
            data_ptr[j] /= sum;
    }
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Mean(DTensor<CPU, Dtype>& a, int axis)
{
	ASSERT(axis == -1, "currently only support global mean");
	Reshape({1});

	Dtype s = 0;
	for (size_t i = 0; i < a.shape.Count(); ++i)
		s += a.data->ptr[i];
	data->ptr[0] = s / a.shape.Count();
}

template class TensorTemplate<CPU, DENSE, float>;
template class TensorTemplate<CPU, DENSE, double>;

///================================ int tensor ===================================

TensorTemplate<CPU, DENSE, int>::TensorTemplate() : data(nullptr)
{

}

void TensorTemplate<CPU, DENSE, int>::Reshape(std::vector<size_t> l)
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

void TensorTemplate<CPU, DENSE, int>::ShallowCopy(DTensor<CPU, int>& src)
{
	this->shape = src.shape;
	this->data = src.data;
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