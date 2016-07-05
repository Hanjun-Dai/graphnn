#include "sparse_matrix.h"
#include <algorithm>
#include <cstring>
#include "mkl_helper.h"

template<typename Dtype>
SparseMat<CPU, Dtype>::SparseMat()
{
		this->count = this->rows = this->cols = 0;
		data = std::make_shared< SpData<CPU, Dtype> >();
}

template<typename Dtype>
SparseMat<CPU, Dtype>::SparseMat(size_t _rows, size_t _cols)
{
		this->rows = _rows; this->cols = _cols;
		this->count = this->rows * this->cols;
		data = std::make_shared< SpData<CPU, Dtype> >();
}

template<typename Dtype>
SparseMat<CPU, Dtype>::~SparseMat()
{	
			
}

template<typename Dtype>
void SparseMat<CPU, Dtype>::Serialize(FILE* fid)
{
		IMatrix<CPU, Dtype>::Serialize(fid);
		data->Serialize(fid);
}

template<typename Dtype>
void SparseMat<CPU, Dtype>::Deserialize(FILE* fid)
{
		IMatrix<CPU, Dtype>::Deserialize(fid);
		data = std::make_shared< SpData<CPU, Dtype> >();
		data->Deserialize(fid);
}
		
template<typename Dtype>
void SparseMat<CPU, Dtype>::Resize(size_t newRos, size_t newCols)
{
		this->count = newRos * newCols;
		this->rows = newRos;
		this->cols = newCols;
}

template<typename Dtype>
void SparseMat<CPU, Dtype>::Print2Screen() // debug only
{
        std::cerr << "========mat content========" << std::endl;
        for (size_t i = 0; i < this->rows; ++i)
        {
            for (int j = data->ptr[i]; j < data->ptr[i + 1]; ++j)
                std::cerr << "(" << i << "," << data->col_idx[j] << ") : " << data->val[j] << std::endl;
        }
        std::cerr << "========    end    ========" << std::endl;    
}

template<typename Dtype>
void SparseMat<CPU, Dtype>::ResizeSp(int newNNZ, int newNPtr)
{
		if (newNNZ > data->nzCap || newNPtr > data->ptrCap)
		{
			if (newNNZ > data->nzCap)
				data->nzCap = std::max(newNNZ, data->nzCap * 2);
			if (newNPtr > data->ptrCap)
				data->ptrCap = std::max(newNPtr, data->ptrCap * 2);
				data = std::make_shared< SpData<CPU, Dtype> >(data->nzCap, data->ptrCap);
		}
		data->nnz = newNNZ;
		data->len_ptr = newNPtr;
}

template<typename Dtype>
Dtype SparseMat<CPU, Dtype>::Asum()
{
    return MKLHelper_Asum(data->nnz, data->val);
}

template<typename Dtype>
void SparseMat<CPU, Dtype>::CopyFrom(SparseMat<CPU, Dtype>& src)
{				
		this->rows = src.rows;
		this->cols = src.cols;
		this->count = src.count;
		ResizeSp(src.data->nnz, src.data->len_ptr);
		memcpy(data->val, src.data->val, sizeof(Dtype) * src.data->nnz);
		memcpy(data->col_idx, src.data->col_idx, sizeof(int) * src.data->nnz);
		memcpy(data->ptr, src.data->ptr, sizeof(int) * src.data->len_ptr);						
}

template<typename Dtype>
void SparseMat<CPU, Dtype>::CopyFrom(SparseMat<GPU, Dtype>& src)
{				
		this->rows = src.rows;
		this->cols = src.cols;
		this->count = src.count;
		ResizeSp(src.data->nnz, src.data->len_ptr);
		cudaMemcpyAsync(data->val, src.data->val, sizeof(Dtype) * src.data->nnz, cudaMemcpyDeviceToHost, GPUHandle::streams[src.streamid]);	
        cudaMemcpyAsync(data->col_idx, src.data->col_idx, sizeof(int) * src.data->nnz, cudaMemcpyDeviceToHost, GPUHandle::streams[src.streamid]);
		cudaMemcpyAsync(data->ptr, src.data->ptr, sizeof(int) * src.data->len_ptr, cudaMemcpyDeviceToHost, GPUHandle::streams[src.streamid]);
}

template class SparseMat<CPU, double>;
template class SparseMat<CPU, float>;