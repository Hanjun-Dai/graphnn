#include "tensor/cpu_sparse_tensor.h"
#include "tensor/t_data.h"
#include "tensor/cpu_dense_tensor.h"
#include "util/mem_holder.h"
#include <cstring>
#include <cassert>

#ifdef USE_GPU
#include "tensor/gpu_sparse_tensor.h"
#endif

namespace gnn 
{

template<typename Dtype>
TensorTemplate<CPU, CSR_SPARSE, Dtype>::TensorTemplate() : data(nullptr)
{
}

template<typename Dtype>
void TensorTemplate<CPU, CSR_SPARSE, Dtype>::Reshape(std::vector<size_t> l)
{
	ASSERT(l.size() == 2, "only support sparse matrix");
	this->shape.Reshape(l);
}

template<typename Dtype>
void TensorTemplate<CPU, CSR_SPARSE, Dtype>::ResizeSp(int newNNZ, int newNPtr)
{
	if (this->data == nullptr)
		this->data = std::make_shared< SparseData<CPU, Dtype> >();

	if (newNNZ > data->nzCap || newNPtr > data->ptrCap)
	{
		if (newNNZ > data->nzCap)
			data->nzCap = std::max(newNNZ, data->nzCap * 2);
		if (newNPtr > data->ptrCap)
			data->ptrCap = std::max(newNPtr, data->ptrCap * 2);
		data = std::make_shared< SparseData<CPU, Dtype> >(data->nzCap, data->ptrCap);
	}
	data->nnz = newNNZ;
	data->len_ptr = newNPtr;
}

template<typename Dtype>
MatType TensorTemplate<CPU, CSR_SPARSE, Dtype>::GetMatType()
{
	return MatType::sparse;
}

template<typename Dtype>
MatMode TensorTemplate<CPU, CSR_SPARSE, Dtype>::GetMatMode()
{
	return MatMode::cpu;
}

template<typename Dtype>
void TensorTemplate<CPU, CSR_SPARSE, Dtype>::CopyFrom(SpTensor<CPU, Dtype>& src)
{
	this->shape = src.shape;
	ResizeSp(src.data->nnz, src.data->len_ptr);
	memcpy(data->val, src.data->val, sizeof(Dtype) * src.data->nnz);
	memcpy(data->col_idx, src.data->col_idx, sizeof(int) * src.data->nnz);
	memcpy(data->row_ptr, src.data->row_ptr, sizeof(int) * src.data->len_ptr);
}

#ifdef USE_GPU
template<typename Dtype>
void TensorTemplate<CPU, CSR_SPARSE, Dtype>::CopyFrom(SpTensor<GPU, Dtype>& src)
{
	this->shape = src.shape;
	ResizeSp(src.data->nnz, src.data->len_ptr);
	cudaMemcpy(data->val, src.data->val, sizeof(Dtype) * src.data->nnz, cudaMemcpyDeviceToHost);
	cudaMemcpy(data->col_idx, src.data->col_idx, sizeof(int) * src.data->nnz, cudaMemcpyDeviceToHost);
	cudaMemcpy(data->row_ptr, src.data->row_ptr, sizeof(int) * src.data->len_ptr, cudaMemcpyDeviceToHost);
}
#endif

template<typename Dtype>
void TensorTemplate<CPU, CSR_SPARSE, Dtype>::ShallowCopy(SpTensor<CPU, Dtype>& src)
{
	this->shape = src.shape;
	this->data = src.data;
}

template<typename Dtype>
void TensorTemplate<CPU, CSR_SPARSE, Dtype>::ArgMax(DTensor<CPU, int>& dst, uint axis)
{
	ASSERT(axis == 0, "not supported for axis > 0 in CPU Sparse Tensor");
	dst.Reshape({this->shape[0]});

	for (size_t i = 0; i < this->shape[0]; ++i)
	{
		dst.data->ptr[i] = 0;
		Dtype cur_max = 0;
		for (int j = data->row_ptr[i]; j < data->row_ptr[i + 1]; ++j)
			if (j == data->row_ptr[i] || data->val[j] > cur_max)
			{
				cur_max = data->val[j];
				dst.data->ptr[i] = data->col_idx[j];
			}
	}
}

template class TensorTemplate<CPU, CSR_SPARSE, float>;
template class TensorTemplate<CPU, CSR_SPARSE, double>;

TensorTemplate<CPU, CSR_SPARSE, int>::TensorTemplate() : data(nullptr)
{
}

void TensorTemplate<CPU, CSR_SPARSE, int>::Reshape(std::vector<size_t> l)
{
}

MatType TensorTemplate<CPU, CSR_SPARSE, int>::GetMatType()
{
	return MatType::sparse;
}

MatMode TensorTemplate<CPU, CSR_SPARSE, int>::GetMatMode()
{
	return MatMode::cpu;
}

void TensorTemplate<CPU, CSR_SPARSE, int>::ShallowCopy(SpTensor<CPU, int>& src)
{
	this->shape = src.shape;
	this->data = src.data;
}

void TensorTemplate<CPU, CSR_SPARSE, int>::ResizeSp(int newNNZ, int newNPtr)
{
	if (this->data == nullptr)
		this->data = std::make_shared< SparseData<CPU, int> >();

	if (newNNZ > data->nzCap || newNPtr > data->ptrCap)
	{
		if (newNNZ > data->nzCap)
			data->nzCap = std::max(newNNZ, data->nzCap * 2);
		if (newNPtr > data->ptrCap)
			data->ptrCap = std::max(newNPtr, data->ptrCap * 2);
		data = std::make_shared< SparseData<CPU, int> >(data->nzCap, data->ptrCap);
	}
	data->nnz = newNNZ;
	data->len_ptr = newNPtr;
}

template class TensorTemplate<CPU, CSR_SPARSE, int>;

}

