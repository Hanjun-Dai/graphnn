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
	row_idxes.Reshape({0});
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::ReshapeLike(RowSpTensor<GPU, Dtype>& src)
{
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseCopy(DTensor<GPU, Dtype>& src)
{
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Scale(Dtype scalar)
{
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Sqrt()
{
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseAdd(Dtype scalar)
{
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseInv()
{
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::ElewiseMul(DTensor<GPU, Dtype>& src)
{
	throw std::logic_error(std::string("not implemented"));
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
		throw std::logic_error(std::string("not implemented virtual func: "));
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
	throw std::logic_error(std::string("not implemented"));
	//Full().MM(a, b, transA, transB, alpha, beta);
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseAxpy(Dtype a, DTensor<GPU, Dtype>& x)
{
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::RowSparseAxpby(Dtype a, DTensor<GPU, Dtype>& x, Dtype b)
{
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
Dtype TensorTemplate<GPU, ROW_SPARSE, Dtype>::Norm2()
{
	if (is_full)
		return Full().Norm2();
	throw std::logic_error(std::string("not implemented"));
}

template<typename Dtype>
void TensorTemplate<GPU, ROW_SPARSE, Dtype>::Square()
{
	if (is_full)
		Full().Square();
	else {
		throw std::logic_error(std::string("not implemented"));
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