#include "tensor/gpu_sparse_tensor.h"
#include "tensor/cpu_sparse_tensor.h"
#include "tensor/t_data.h"
#include "tensor/gpu_dense_tensor.h"
#include "tensor/gpu_reduce_kernel.h"
#include "util/mem_holder.h"
#include <cstring>
#include <cassert>

namespace gnn
{

template<typename Dtype>
TensorTemplate<GPU, CSR_SPARSE, Dtype>::TensorTemplate() : data(nullptr)
{
}

template<typename Dtype>
void TensorTemplate<GPU, CSR_SPARSE, Dtype>::Reshape(std::vector<size_t> l)
{
	ASSERT(l.size() == 2, "only support sparse matrix");
	this->shape.Reshape(l);
}

template<typename Dtype>
void TensorTemplate<GPU, CSR_SPARSE, Dtype>::ResizeSp(int newNNZ, int newNPtr)
{
	if (this->data == nullptr)
		this->data = std::make_shared< SparseData<GPU, Dtype> >();

	if (newNNZ > data->nzCap || newNPtr > data->ptrCap)
	{
		if (newNNZ > data->nzCap)
			data->nzCap = std::max(newNNZ, data->nzCap * 2);
		if (newNPtr > data->ptrCap)
			data->ptrCap = std::max(newNPtr, data->ptrCap * 2);
		data = std::make_shared< SparseData<GPU, Dtype> >(data->nzCap, data->ptrCap);
	}
	data->nnz = newNNZ;
	data->len_ptr = newNPtr;
}

template<typename Dtype>
MatType TensorTemplate<GPU, CSR_SPARSE, Dtype>::GetMatType()
{
	return MatType::sparse;
}

template<typename Dtype>
MatMode TensorTemplate<GPU, CSR_SPARSE, Dtype>::GetMatMode()
{
	return MatMode::gpu;
}

template<typename Dtype>
void TensorTemplate<GPU, CSR_SPARSE, Dtype>::CopyFrom(SpTensor<CPU, Dtype>& src)
{
	this->shape = src.shape;
	ResizeSp(src.data->nnz, src.data->len_ptr);
	cudaMemcpy(data->val, src.data->val, sizeof(Dtype) * src.data->nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(data->col_idx, src.data->col_idx, sizeof(int) * src.data->nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(data->row_ptr, src.data->row_ptr, sizeof(int) * src.data->len_ptr, cudaMemcpyHostToDevice);
}

template<typename Dtype>
void TensorTemplate<GPU, CSR_SPARSE, Dtype>::CopyFrom(SpTensor<GPU, Dtype>& src)
{
	this->shape = src.shape;
	ResizeSp(src.data->nnz, src.data->len_ptr);
	cudaMemcpy(data->val, src.data->val, sizeof(Dtype) * src.data->nnz, cudaMemcpyDeviceToDevice);
	cudaMemcpy(data->col_idx, src.data->col_idx, sizeof(int) * src.data->nnz, cudaMemcpyDeviceToDevice);
	cudaMemcpy(data->row_ptr, src.data->row_ptr, sizeof(int) * src.data->len_ptr, cudaMemcpyDeviceToDevice);
}

template<typename Dtype>
void TensorTemplate<GPU, CSR_SPARSE, Dtype>::ShallowCopy(SpTensor<GPU, Dtype>& src)
{
	this->shape = src.shape;
	this->data = src.data;
}

template<typename dstDtype, typename srcDtype>
__global__ void SparseMatColReduceKernel(dstDtype* dst, int* row_ptr, int* col_idx, srcDtype *val)
{
    __shared__ dstDtype buffer[REDUCE_THREADS];

    int i_start = row_ptr[blockIdx.x] + threadIdx.x;
    int i_end = row_ptr[blockIdx.x + 1];
    int i_step = blockDim.x;
    if (i_start < i_end)
    	buffer[threadIdx.x] = i_start;
    for (int i = i_start + i_step; i < i_end; i += i_step)
    {
    	if (val[i] > val[buffer[threadIdx.x]])
    		buffer[threadIdx.x] = i;
    }
    __syncthreads();

    int shift;
    for (int i = REDUCE_THREAD_BITS - 1; i >= 0; --i)
    {
    	shift = 1 << i;
    	if (threadIdx.x < shift && threadIdx.x + shift < row_ptr[blockIdx.x + 1] - row_ptr[blockIdx.x])
    	{
    		if (val[buffer[threadIdx.x]] < buffer[threadIdx.x + shift])
    			buffer[threadIdx.x] = buffer[threadIdx.x + shift];
    	}
		__syncthreads();
    }
    if (threadIdx.x == 0)
    	dst[blockIdx.x] = col_idx[buffer[0]];
}

template<typename Dtype>
void TensorTemplate<GPU, CSR_SPARSE, Dtype>::ArgMax(DTensor<GPU, int>& dst, uint axis)
{
	ASSERT(axis == 0, "not supported for axis > 0 in GPU Sparse Tensor");
	dst.Reshape({this->shape[0]});
	dim3 blocks(this->shape[0]);
	dim3 threads(REDUCE_THREADS);
    SparseMatColReduceKernel<<<blocks, threads, 0, cudaStreamPerThread>>> (dst.data->ptr, data->row_ptr, data->col_idx, data->val);
}

template class TensorTemplate<GPU, CSR_SPARSE, float>;
template class TensorTemplate<GPU, CSR_SPARSE, double>;

TensorTemplate<GPU, CSR_SPARSE, int>::TensorTemplate() : data(nullptr)
{
}

void TensorTemplate<GPU, CSR_SPARSE, int>::Reshape(std::vector<size_t> l)
{
}

MatType TensorTemplate<GPU, CSR_SPARSE, int>::GetMatType()
{
	return MatType::sparse;
}

MatMode TensorTemplate<GPU, CSR_SPARSE, int>::GetMatMode()
{
	return MatMode::gpu;
}

void TensorTemplate<GPU, CSR_SPARSE, int>::ShallowCopy(SpTensor<GPU, int>& src)
{
	this->shape = src.shape;
	this->data = src.data;
}

void TensorTemplate<GPU, CSR_SPARSE, int>::ResizeSp(int newNNZ, int newNPtr)
{
	if (this->data == nullptr)
		this->data = std::make_shared< SparseData<GPU, int> >();

	if (newNNZ > data->nzCap || newNPtr > data->ptrCap)
	{
		if (newNNZ > data->nzCap)
			data->nzCap = std::max(newNNZ, data->nzCap * 2);
		if (newNPtr > data->ptrCap)
			data->ptrCap = std::max(newNPtr, data->ptrCap * 2);
		data = std::make_shared< SparseData<GPU, int> >(data->nzCap, data->ptrCap);
	}
	data->nnz = newNNZ;
	data->len_ptr = newNPtr;
}

template class TensorTemplate<GPU, CSR_SPARSE, int>;

}