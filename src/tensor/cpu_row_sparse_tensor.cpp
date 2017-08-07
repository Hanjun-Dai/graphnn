#include "tensor/cpu_row_sparse_tensor.h"
#include "tensor/t_data.h"
#include "tensor/cpu_dense_tensor.h"
#include "tensor/cpu_sparse_tensor.h"
#include "tensor/mkl_helper.h"
#include "util/mem_holder.h"
#include "tbb/tbb.h"
#include "fmt/format.h"
#include <cstring>
#include <cassert>

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
	row_idxes.Reshape({this->shape.dims[0]});
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
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];
			memset(data->ptr + row_idx * dim, 0, sizeof(Dtype) * dim);
		});
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
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::SparseMM(SpTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{
	ASSERT(transA == Trans::T, "only for bp right now");

	if (is_full)
	{
		Full().MM(a, b, transA, transB, alpha, beta);
		return;
	}

	size_t cur_cnt = row_idxes.shape.Count();
	idx_buf.Reshape({a.data->nnz + cur_cnt});
	memcpy(idx_buf.data->ptr, row_idxes.data->ptr, sizeof(int) * cur_cnt);
	memcpy(idx_buf.data->ptr + cur_cnt, a.data->col_idx, sizeof(int) * a.data->nnz);
	std::sort(idx_buf.data->ptr, idx_buf.data->ptr + idx_buf.shape.Count());

	auto last = std::unique(idx_buf.data->ptr, idx_buf.data->ptr + idx_buf.shape.Count());
	cur_cnt = last - idx_buf.data->ptr;

	row_idxes.Reshape({cur_cnt});
	memcpy(row_idxes.data->ptr, idx_buf.data->ptr, sizeof(int) * cur_cnt);

	Full().MM(a, b, transA, transB, alpha, beta);
	is_full = false;
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::Axpy(Dtype a, DTensor<CPU, Dtype>& x)
{
	Full().Axpy(a, x);
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::RowSparseAxpy(Dtype a, DTensor<CPU, Dtype>& x)
{
	if (is_full)
		Full().Axpy(a, x);
	else if (row_idxes.shape.Count())
	{
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];
			MKL_Axpy(dim, a, 
					x.data->ptr + row_idx * dim, 
					data->ptr + row_idx * dim);
		});
	}
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
	else if (row_idxes.shape.Count())
		throw std::logic_error(std::string("not implemented"));
	else 
		return 0;
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::Square()
{
	if (is_full)
		Full().Square();
	else if (row_idxes.shape.Count()) {
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

