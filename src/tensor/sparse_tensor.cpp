#include "tensor/sparse_tensor.h"
#include "tensor/t_data.h"
#include "util/mem_holder.h"
#include <cstring>
#include <cassert>

namespace gnn 
{

template<typename Dtype>
TensorTemplate<CPU, SPARSE, Dtype>::TensorTemplate() : data(nullptr)
{
}

template<typename Dtype>
void TensorTemplate<CPU, SPARSE, Dtype>::Reshape(std::initializer_list<uint> l)
{
}

template<typename Dtype>
MatType TensorTemplate<CPU, SPARSE, Dtype>::GetMatType()
{
	return MatType::sparse;
}

template<typename Dtype>
MatMode TensorTemplate<CPU, SPARSE, Dtype>::GetMatMode()
{
	return MatMode::cpu;
}

template<typename Dtype>
void TensorTemplate<CPU, SPARSE, Dtype>::CopyFrom(SpTensor<CPU, Dtype>& src)
{
}

template<typename Dtype>
void TensorTemplate<CPU, SPARSE, Dtype>::ResizeSp(int newNNZ, int newNPtr)
{

}

template class TensorTemplate<CPU, SPARSE, float>;
template class TensorTemplate<CPU, SPARSE, double>;

}

