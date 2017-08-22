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
	row_idxes.Reshape({0});
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);
	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<GPU, Dtype> >();

	this->data->Resize(this->shape.Count());
	is_full = true;
	row_idxes.Reshape({this->shape.dims[0]});
	row_idxes.Reshape({0});
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::ReshapeLike(RowSpTensor<GPU, Dtype>& src)
{
	ASSERT(!is_full && row_idxes.shape.Count() == 0, "should be empty before reshaping");
	ASSERT(data && data->mem_size >= src.shape.Count(), "should manually allocate memory before reshaping");

	this->shape.Reshape(src.shape.dims);
	this->is_full = src.is_full;
	if (!src.is_full)
		row_idxes.CopyFrom(src.row_idxes);
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseCopy(DTensor<GPU, Dtype>& src)
{
	if (is_full)
		Full().CopyFrom(src);
	else if (row_idxes.shape.Count())
	{
		throw std::logic_error(std::string("RowSparseCopy not implemented"));
	}	
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Scale(Dtype scalar)
{
	if (is_full)
		Full().Scale(scalar);
	else if (row_idxes.shape.Count())
	{
		throw std::logic_error(std::string("Scale not implemented"));
	}	
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Sqrt()
{
	if (is_full)
		Full().Sqrt();
	else if (row_idxes.shape.Count())
	{
		throw std::logic_error(std::string("Sqrt not implemented"));
	}	
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseAdd(Dtype scalar)
{
	if (is_full)
		Full().Add(scalar);
	else if (row_idxes.shape.Count())
	{
		throw std::logic_error(std::string("RowSparseAdd not implemented"));
	}	
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseInv()
{
	if (is_full)
		Full().Inv();
	else if (row_idxes.shape.Count())
	{
		throw std::logic_error(std::string("RowSparseInv not implemented"));
	}	
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::ElewiseMul(DTensor<GPU, Dtype>& src)
{
	ASSERT(this->shape == src.shape, "shape doesn't match in ElewiseMul");

	if (is_full)
		Full().ElewiseMul(src);
	else if (row_idxes.shape.Count())
	{
		throw std::logic_error(std::string("ElewiseMul not implemented"));
	}		
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
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSpZeros()
{
	if (is_full)
		Full().Zeros();
	else if (row_idxes.shape.Count()) {
		throw std::logic_error(std::string("RowSpZeros not implemented"));
	}
	row_idxes.Reshape({0});
	is_full = false;
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::FullZeros()
{
	is_full = true;
	RowSpZeros();
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseFill(Dtype scalar)
{
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::SparseMM(SpTensor<GPU, Dtype>& a, DTensor<GPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{
	ASSERT(transA == Trans::T, "only for bp right now");

	if (is_full)
	{
		Full().MM(a, b, transA, transB, alpha, beta);
		return;
	}	
	throw std::logic_error(std::string("SparseMM not implemented"));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseAxpy(Dtype a, DTensor<GPU, Dtype>& x)
{
	if (is_full)
		Full().Axpy(a, x);
	else if (row_idxes.shape.Count())
	{
		throw std::logic_error(std::string("RowSparseAxpy not implemented"));
	}	
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseAxpby(Dtype a, DTensor<GPU, Dtype>& x, Dtype b)
{
	if (is_full)
		Full().Axpby(a, x, b);
	else if (row_idxes.shape.Count())
	{
		throw std::logic_error(std::string("RowSparseAxpby not implemented"));
	}
}

template<typename Dtype>
Dtype TensorTemplate<GPU, ROW_SPARSE, Dtype>::Norm2()
{
	if (is_full)
		return Full().Norm2();
	else if (row_idxes.shape.Count())
	{
		throw std::logic_error(std::string("Norm2 not implemented"));
	}
	else 
		return 0;
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Square()
{
	if (is_full)
		Full().Square();
	else if (row_idxes.shape.Count()) {
		throw std::logic_error(std::string("Square not implemented"));
	}
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
	return MatType::row_sparse;
}

MatMode TensorTemplate<GPU, ROW_SPARSE, int>::GetMatMode()
{
	return MatMode::gpu;
}

template class TensorTemplate<GPU, ROW_SPARSE, int>;

}