#include "sparse_matrix.h"
#include "cuda_helper.h"
#include <algorithm>

template<typename Dtype>
SparseMat<GPU, Dtype>::SparseMat()
{
		this->count = this->rows = this->cols = 0;
		streamid = 0;
		data = std::make_shared< SpData<GPU, Dtype> >();
		cusparseCreateMatDescr(&descr);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
}

template<typename Dtype>
SparseMat<GPU, Dtype>::SparseMat(size_t _rows, size_t _cols, unsigned _streamid)
{
		this->rows = _rows; this->cols = _cols;
		this->count = _rows * _cols;
		streamid = _streamid;
		data = std::make_shared< SpData<GPU, Dtype> >();
		cusparseCreateMatDescr(&descr);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
}

template<typename Dtype>
SparseMat<GPU, Dtype>::~SparseMat()
{
		cusparseDestroyMatDescr(descr);
}

template<typename Dtype>
void SparseMat<GPU, Dtype>::Resize(size_t newRos, size_t newCols)
{
		this->count = newRos * newCols;
		this->rows = newRos;
		this->cols = newCols;
}

template<typename Dtype>
void SparseMat<GPU, Dtype>::Print2Screen() // debug only
{
        throw "not implemented";
}

template<typename Dtype>
void SparseMat<GPU, Dtype>::ResizeSp(int newNNZ, int newNPtr)
{
		if (newNNZ > data->nzCap || newNPtr > data->ptrCap)
		{
			if (newNNZ > data->nzCap)
				data->nzCap = std::max(newNNZ, data->nzCap * 2);
			if (newNPtr > data->ptrCap)
				data->ptrCap = std::max(newNPtr, data->ptrCap * 2);
				data = std::make_shared< SpData<GPU, Dtype> >(data->nzCap, data->ptrCap);
		}
		data->nnz = newNNZ;
		data->len_ptr = newNPtr;
}
	
template<typename Dtype>
Dtype SparseMat<GPU, Dtype>::Asum()
{
		return CudaHelper_Asum(GPUHandle::cublashandle, data->nnz, data->val);
}
	
template<typename Dtype>
void SparseMat<GPU, Dtype>::CopyFrom(SparseMat<CPU, Dtype>& src)
{				
		this->rows = src.rows;
		this->cols = src.cols;
		this->count = src.count;
		ResizeSp(src.data->nnz, src.data->len_ptr);
		cudaMemcpyAsync(data->val, src.data->val, sizeof(Dtype) * src.data->nnz, cudaMemcpyHostToDevice, GPUHandle::streams[streamid]);	
        cudaMemcpyAsync(data->col_idx, src.data->col_idx, sizeof(int) * src.data->nnz, cudaMemcpyHostToDevice, GPUHandle::streams[streamid]);
		cudaMemcpyAsync(data->ptr, src.data->ptr, sizeof(int) * src.data->len_ptr, cudaMemcpyHostToDevice, GPUHandle::streams[streamid]);
}

template<typename Dtype>
void SparseMat<GPU, Dtype>::CopyFrom(SparseMat<GPU, Dtype>& src)
{				
		this->rows = src.rows;
		this->cols = src.cols;
		this->count = src.count;
		ResizeSp(src.data->nnz, src.data->len_ptr);
		cudaMemcpyAsync(data->val, src.data->val, sizeof(Dtype) * src.data->nnz, cudaMemcpyDeviceToDevice, GPUHandle::streams[streamid]);	
        cudaMemcpyAsync(data->col_idx, src.data->col_idx, sizeof(int) * src.data->nnz, cudaMemcpyDeviceToDevice, GPUHandle::streams[streamid]);
		cudaMemcpyAsync(data->ptr, src.data->ptr, sizeof(int) * src.data->len_ptr, cudaMemcpyDeviceToDevice, GPUHandle::streams[streamid]);
}

template<typename Dtype>
void SparseMat<GPU, Dtype>::Serialize(FILE* fid)
{
		IMatrix<GPU, Dtype>::Serialize(fid);
		data->Serialize(fid);
}

template<typename Dtype>
void SparseMat<GPU, Dtype>::Deserialize(FILE* fid)
{
		IMatrix<GPU, Dtype>::Deserialize(fid);
		data = std::make_shared< SpData<GPU, Dtype> >();
		data->Deserialize(fid);
}

template class SparseMat<GPU, double>;
template class SparseMat<GPU, float>;