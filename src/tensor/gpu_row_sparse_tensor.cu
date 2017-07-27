#include "tensor/gpu_row_sparse_tensor.h"
#include "tensor/cpu_sparse_tensor.h"
#include "tensor/t_data.h"
#include "tensor/gpu_dense_tensor.h"
#include "util/mem_holder.h"
#include <cstring>
#include <cassert>

namespace gnn
{

template<typename Dtype>
TensorTemplate<GPU, ROW_SPARSE, Dtype>::TensorTemplate() : data(nullptr), is_full(false)
{
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);
	is_full = false;
	row_idxes.Reshape({0});
}

template<typename Dtype>
MatType TensorTemplate<GPU, ROW_SPARSE, Dtype>::GetMatType()
{
	return MatType::row_sparse;
}

template<typename Dtype>
MatMode TensorTemplate<GPU, ROW_SPARSE, Dtype>::GetMatMode()
{
	return MatMode::gpu;
}

template<typename Dtype>
DTensor<GPU, Dtype> TensorTemplate<GPU, ROW_SPARSE, Dtype>::Full()
{
	is_full = true;
	return DTensor<GPU, Dtype>(this->shape, this->data->ptr);
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::SparseZeros()
{
	if (is_full)
		Full().Zeros();
	else {
		throw std::logic_error(std::string("not implemented virtual func: "));
	}
	row_idxes.Reshape({0});
	is_full = false;
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::FullZeros()
{
	is_full = true;
	SparseZeros();
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Fill(Dtype scalar)
{
	throw std::logic_error(std::string("not implemented virtual func: "));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::MM(DTensor<GPU, Dtype>& a, DTensor<GPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{
	throw std::logic_error(std::string("not implemented virtual func: "));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::MM(SpTensor<GPU, Dtype>& a, DTensor<GPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{
	throw std::logic_error(std::string("not implemented virtual func: "));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Axpy(Dtype a, DTensor<GPU, Dtype>& x)
{
	throw std::logic_error(std::string("not implemented virtual func: "));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Axpby(Dtype a, DTensor<GPU, Dtype>& x, Dtype b)
{
	throw std::logic_error(std::string("not implemented virtual func: "));
}

template<typename Dtype>
Dtype TensorTemplate<GPU, ROW_SPARSE, Dtype>::Norm2()
{
	throw std::logic_error(std::string("not implemented virtual func: "));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Square()
{
	throw std::logic_error(std::string("not implemented virtual func: "));
}


template class TensorTemplate<GPU, ROW_SPARSE, float>;
template class TensorTemplate<GPU, ROW_SPARSE, double>;


TensorTemplate<GPU, ROW_SPARSE, int>::TensorTemplate()
{
}

void TensorTemplate<GPU, ROW_SPARSE, int>::Reshape(std::vector<size_t> l)
{
}

MatType TensorTemplate<GPU, ROW_SPARSE, int>::GetMatType()
{
	return MatType::sparse;
}

MatMode TensorTemplate<GPU, ROW_SPARSE, int>::GetMatMode()
{
	return MatMode::gpu;
}

template class TensorTemplate<GPU, ROW_SPARSE, int>;

}