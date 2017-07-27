#include "tensor/cpu_row_sparse_tensor.h"
#include "tensor/t_data.h"
#include "tensor/cpu_dense_tensor.h"
#include "util/mem_holder.h"
#include <cstring>
#include <cassert>

#ifdef USE_GPU
#include "tensor/gpu_row_sparse_tensor.h"
#endif

namespace gnn 
{

template<typename Dtype>
TensorTemplate<CPU, ROW_SPARSE, Dtype>::TensorTemplate() : data(nullptr), is_full(false)
{
	row_idxes.Reshape({0});
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);
	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<CPU, Dtype> >();

	this->data->Resize(this->shape.Count());
	is_full = true;
	row_idxes.Reshape({0});
}

template<typename Dtype>
MatType TensorTemplate<CPU, ROW_SPARSE, Dtype>::GetMatType()
{
	return MatType::row_sparse;
}

template<typename Dtype>
MatMode TensorTemplate<CPU, ROW_SPARSE, Dtype>::GetMatMode()
{
	return MatMode::cpu;
}

template<typename Dtype>
DTensor<CPU, Dtype> TensorTemplate<CPU, ROW_SPARSE, Dtype>::Full()
{
	is_full = true;
	return DTensor<CPU, Dtype>(this->shape, this->data->ptr);
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::SparseZeros()
{
	if (is_full)
		Full().Zeros();
	else if (row_idxes.shape.Count()) 
	{
		throw std::logic_error(std::string("not implemented"));
	}
	row_idxes.Reshape({0});
	is_full = false;
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::FullZeros()
{
	is_full = true;
	SparseZeros();
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::Fill(Dtype scalar)
{
	Full().Fill(scalar);
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::MM(DTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{
	Full().MM(a, b, transA, transB, alpha, beta);
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::MM(SpTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{
	Full().MM(a, b, transA, transB, alpha, beta);
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::Axpy(Dtype a, DTensor<CPU, Dtype>& x)
{
	Full().Axpy(a, x);
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::Axpby(Dtype a, DTensor<CPU, Dtype>& x, Dtype b)
{
	Full().Axpby(a, x, b);
}

template<typename Dtype>
Dtype TensorTemplate<CPU, ROW_SPARSE, Dtype>::Norm2()
{
	if (is_full)
		return Full().Norm2();
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::Square()
{
	if (is_full)
		Full().Square();
	else {
		throw std::logic_error(std::string("not implemented"));
	}
}

template class TensorTemplate<CPU, ROW_SPARSE, float>;
template class TensorTemplate<CPU, ROW_SPARSE, double>;

TensorTemplate<CPU, ROW_SPARSE, int>::TensorTemplate()
{
}

void TensorTemplate<CPU, ROW_SPARSE, int>::Reshape(std::vector<size_t> l)
{
}

MatType TensorTemplate<CPU, ROW_SPARSE, int>::GetMatType()
{
	return MatType::row_sparse;
}

MatMode TensorTemplate<CPU, ROW_SPARSE, int>::GetMatMode()
{
	return MatMode::cpu;
}

template class TensorTemplate<CPU, ROW_SPARSE, int>;

}

