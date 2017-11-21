#include "tensor/cpu_row_sparse_tensor.h"
#include "tensor/t_data.h"
#include "tensor/cpu_dense_tensor.h"
#include "tensor/cpu_sparse_tensor.h"
#include "tensor/mkl_helper.h"
#include "util/mem_holder.h"
#include "tbb/tbb.h"
#include <cstring>
#include <cmath>
#include <atomic>
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
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::ReshapeLike(RowSpTensor<CPU, Dtype>& src)
{
	ASSERT(!is_full && row_idxes.shape.Count() == 0, "should be empty before reshaping");
	ASSERT(data && data->mem_size >= src.shape.Count(), "should manually allocate memory before reshaping");

	this->shape.Reshape(src.shape.dims);
	this->is_full = src.is_full;
	if (!src.is_full)
		row_idxes.CopyFrom(src.row_idxes);
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::RowSparseCopy(DTensor<CPU, Dtype>& src)
{
	if (is_full)
		Full().CopyFrom(src);
	else if (row_idxes.shape.Count())
	{
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];

			memcpy(data->ptr + row_idx * dim, 
					src.data->ptr + row_idx * dim, 
					sizeof(Dtype) * dim);
		});
	}
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::Scale(Dtype scalar)
{
	if (is_full)
		Full().Scale(scalar);
	else if (row_idxes.shape.Count())
	{
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];
			Dtype* cur_ptr = data->ptr + row_idx * dim;
			for (size_t i = 0; i < dim; ++i)
				cur_ptr[i] *= scalar;
		});
	}
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::Sqrt()
{
	if (is_full)
		Full().Sqrt();
	else if (row_idxes.shape.Count())
	{
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];
			MKL_Sqrt(dim, data->ptr + row_idx * dim, data->ptr + row_idx * dim);
		});
	}
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::RowSparseAdd(Dtype scalar)
{
	if (is_full)
		Full().Add(scalar);
	else if (row_idxes.shape.Count())
	{
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];
			Dtype* cur_ptr = data->ptr + row_idx * dim;
			for (size_t i = 0; i < dim; ++i)
				cur_ptr[i] += scalar;
		});
	}
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::RowSparseInv()
{
	if (is_full)
		Full().Inv();
	else if (row_idxes.shape.Count())
	{
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];
			MKL_Inv(dim, data->ptr + row_idx * dim, data->ptr + row_idx * dim);
		});
	}
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::ElewiseMul(DTensor<CPU, Dtype>& src)
{
	ASSERT(this->shape == src.shape, "shape doesn't match in ElewiseMul");

	if (is_full)
		Full().ElewiseMul(src);
	else if (row_idxes.shape.Count())
	{
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];

			MKL_Mul(dim, 
					src.data->ptr + dim * row_idx, 
					this->data->ptr + dim * row_idx, 
					this->data->ptr + dim * row_idx);
		});
	}
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
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::RowSpZeros()
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
	RowSpZeros();
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::RowSparseFill(Dtype scalar)
{
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::InsertRowIdxes(size_t cnt, int* new_idxes)
{
	// merge
	size_t cur_cnt = row_idxes.shape.Count();
	idx_buf.Reshape({cnt + cur_cnt});
	memcpy(idx_buf.data->ptr, row_idxes.data->ptr, sizeof(int) * cur_cnt);
	memcpy(idx_buf.data->ptr + cur_cnt, new_idxes, sizeof(int) * cnt);

	// unique
	std::sort(idx_buf.data->ptr, idx_buf.data->ptr + idx_buf.shape.Count());
	auto last = std::unique(idx_buf.data->ptr, idx_buf.data->ptr + idx_buf.shape.Count());	
	cur_cnt = last - idx_buf.data->ptr;

	// copy
	row_idxes.Reshape({cur_cnt});
	memcpy(row_idxes.data->ptr, idx_buf.data->ptr, sizeof(int) * cur_cnt);
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

	InsertRowIdxes(a.data->nnz, a.data->col_idx);

	Full().MM(a, b, transA, transB, alpha, beta);
	is_full = false;
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
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::RowSparseAxpby(Dtype a, DTensor<CPU, Dtype>& x, Dtype b)
{
	if (is_full)
		Full().Axpby(a, x, b);
	else if (row_idxes.shape.Count())
	{
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];
			MKL_Axpby(dim, a, 
					x.data->ptr + row_idx * dim, 
					b,
					data->ptr + row_idx * dim);
		});
	}
}

template<typename Dtype>
Dtype TensorTemplate<CPU, ROW_SPARSE, Dtype>::Norm2()
{
	if (is_full)
		return Full().Norm2();
	else if (row_idxes.shape.Count())
	{
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);

		Dtype total_norm = 0.0;
		tbb::mutex ll;
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];
			auto norm = MKL_Norm2(dim, data->ptr + row_idx * dim);
			norm = norm * norm;
			ll.lock();
			total_norm += norm;
			ll.unlock();
		});
		return sqrt(total_norm);
	}
	else 
		return 0;
}

template<typename Dtype>
void TensorTemplate<CPU, ROW_SPARSE, Dtype>::Square()
{
	if (is_full)
		Full().Square();
	else if (row_idxes.shape.Count()) {
		size_t row_cnt = row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = row_idxes.data->ptr[i];
			MKL_Square(dim, data->ptr + row_idx * dim, data->ptr + row_idx * dim);
		});		
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

