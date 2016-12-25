#include "tensor/gpu_sparse_tensor.h"
#include "tensor/t_data.h"
#include "tensor/gpu_dense_tensor.h"
#include "util/mem_holder.h"
#include <cstring>
#include <cassert>

namespace gnn
{

template<typename Dtype>
TensorTemplate<GPU, SPARSE, Dtype>::TensorTemplate() : data(nullptr)
{
}

template<typename Dtype>
void TensorTemplate<GPU, SPARSE, Dtype>::Reshape(std::vector<size_t> l)
{
	ASSERT(l.size() == 2, "only support sparse matrix");
	this->shape.Reshape(l);
}

template<typename Dtype>
MatType TensorTemplate<GPU, SPARSE, Dtype>::GetMatType()
{
	return MatType::sparse;
}

template<typename Dtype>
MatMode TensorTemplate<GPU, SPARSE, Dtype>::GetMatMode()
{
	return MatMode::gpu;
}

template class TensorTemplate<GPU, SPARSE, float>;
template class TensorTemplate<GPU, SPARSE, double>;

TensorTemplate<GPU, SPARSE, int>::TensorTemplate() : data(nullptr)
{
}

void TensorTemplate<GPU, SPARSE, int>::Reshape(std::vector<size_t> l)
{
}

MatType TensorTemplate<GPU, SPARSE, int>::GetMatType()
{
	return MatType::sparse;
}

MatMode TensorTemplate<GPU, SPARSE, int>::GetMatMode()
{
	return MatMode::gpu;
}

template class TensorTemplate<GPU, SPARSE, int>;

}