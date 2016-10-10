#include "dense_matrix.h"
#include "cuda_rand_kernel.cuh"
#include "cuda_unary_kernel.cuh"
#include "cuda_binary_kernel.cuh"
#include "cuda_helper.h"
#include "sparse_matrix.h"
#include <thrust/extrema.h>
#include <cuda_runtime.h>
#include <iostream>
#define min(x, y) (x < y ? x : y)

template<typename Dtype>
DenseMat<GPU, Dtype>::~DenseMat()
{
        pointer_buf.clear();
        cudaStreamSynchronize(GPUHandle::streams[streamid]);    
	MatUtils<GPU>::DelArr(data);
}

template<typename Dtype>
DenseMat<GPU, Dtype>::DenseMat(unsigned int _streamid)
{
	mem_size = this->count = this->rows = this->cols = 0U;
	streamid = _streamid;
	data = nullptr;
	pointer_buf.clear();
}

template<typename Dtype>
DenseMat<GPU, Dtype>::DenseMat(size_t _rows, size_t _cols, unsigned int _streamid)
{
	this->rows = _rows; 
	this->cols = _cols;
	this->count = _rows * _cols;
	mem_size = this->count + (this->count & 1);
	MatUtils<GPU>::MallocArr(data, sizeof(Dtype) * mem_size);
	cudaMemset(data, 0, sizeof(Dtype) * mem_size); 
        dev_ptr = thrust::device_pointer_cast(data);
        pointer_buf.clear();	
	streamid = _streamid;
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Resize(size_t _newRows, size_t _newCols)
{
        cudaStreamSynchronize(GPUHandle::streams[streamid]);    
	this->rows = _newRows; 
	this->cols = _newCols;
	this->count = this->rows * this->cols;
	if (this->count > mem_size)
	{
		mem_size = this->count + (this->count & 1);
		MatUtils<GPU>::DelArr(data);
		MatUtils<GPU>::MallocArr(data, sizeof(Dtype) * mem_size);
                dev_ptr = thrust::device_pointer_cast(data);
                cudaMemset(data, 0, sizeof(Dtype) * mem_size);
	}
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::CopyFrom(DenseMat<CPU, Dtype>& src)
{
	Resize(src.rows, src.cols);    
	cudaMemcpyAsync(data, src.data, sizeof(Dtype) * this->count, cudaMemcpyHostToDevice, GPUHandle::streams[streamid]);    
}
		
template<typename Dtype>
void DenseMat<GPU, Dtype>::CopyFrom(DenseMat<GPU, Dtype>& src)
{
	Resize(src.rows, src.cols);
	cudaMemcpyAsync(data, src.data, sizeof(Dtype) * this->count, cudaMemcpyDeviceToDevice, GPUHandle::streams[streamid]);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::CopyFrom(SparseMat<CPU, Dtype>& src)
{
	Resize(src.rows, src.cols);
	throw "not implemented";
	//memcpy(data, src.data, sizeof(Dtype) * this->count);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::CopyFrom(SparseMat<GPU, Dtype>& src)
{
	Resize(src.rows, src.cols);
	throw "not implemented";		
	//memcpy(data, src.data, sizeof(Dtype) * this->count);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Zeros(size_t _rows, size_t _cols)
{
	Resize(_rows, _cols);
	cudaMemset(data, 0, this->count * sizeof(Dtype));
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Zeros()
{
    if (this->count)
	   cudaMemset(data, 0, this->count * sizeof(Dtype));
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::SetRandN(Dtype mean, Dtype std, size_t _rows, size_t _cols)
{
	if (_rows && _cols)
	{
		Resize(_rows, _cols);
	}
    SetRand(this->data, this->count, NormalRandomizer<Dtype>(mean, std), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::SetRandU(Dtype lb, Dtype ub, size_t _rows, size_t _cols)
{
        if (_rows && _cols)
	{
		Resize(_rows, _cols);
    	}
        SetRand(this->data, this->count, UniformRandomizer<Dtype>(lb, ub), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::SetRandSign(size_t _rows, size_t _cols)
{
	if (_rows && _cols)
	{
		Resize(_rows, _cols);
	}
	SetRand(this->data, this->count, BinomialRandomizer<Dtype>(), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::SetRandChi2(Dtype degree, size_t _rows, size_t _cols)
{
	if (_rows && _cols)
	{
		Resize(_rows, _cols);
	}
	SetRand(this->data, this->count, ChisquareRandomizer<Dtype>(degree), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Add(Dtype scalar)
{
        UnaryOp(this->data, this->count, UnaryAdd<Dtype>(scalar), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Fill(Dtype scalar)
{	
        UnaryOp(data, this->count, UnarySet<Dtype>(scalar), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Scale(Dtype scalar)
{	
        if (scalar == 0.0)
        {
            Fill(0);
        } else if (scalar != 1)
		UnaryOp(data, this->count, UnaryScale<Dtype>(scalar), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Power(Dtype scalar)
{
        UnaryOp(this->data, this->count, UnaryPow<Dtype>(scalar), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Inv()
{
        UnaryOp(this->data, this->count, UnaryInv<Dtype>(), streamid); 
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::InvSqrt()
{
        UnaryOp(this->data, this->count, UnaryInvSqrt<Dtype>(), streamid); 
}

// Copied from https://github.com/torch/cunn/blob/master/SoftMax.cu
template<typename Dtype>
__global__ void cunn_SoftMax_updateOutput_kernel(Dtype *orig_ptr, int batch_size, int dim)
{
    __shared__ Dtype buffer[SOFTMAX_THREADS + 1];
    Dtype* dst = orig_ptr + blockIdx.x * dim + blockIdx.y;

    int i_start = threadIdx.x;
    int i_end = dim;
    int i_step = blockDim.x;
    Dtype z;
    // max?
    if (i_start < dim)
        buffer[threadIdx.x] = dst[i_start];
    for (int i = i_start; i < i_end; i += i_step)
    {
        z = dst[i];
        if(buffer[threadIdx.x] < z)
            buffer[threadIdx.x] = z;
    }

    __syncthreads();

    // reduce
    if (threadIdx.x == 0)
    {
        z = buffer[0];
        for (int i = 1; i < min(dim, blockDim.x); i++)
        {
            if(z < buffer[i])
                z = buffer[i];
        }
        buffer[SOFTMAX_THREADS] = z;
    }

    __syncthreads();

    // sum?
    Dtype max_k = buffer[SOFTMAX_THREADS];
    buffer[threadIdx.x] = 0;
    for (int i = i_start; i < i_end; i += i_step) 
    {
        z = cuda_exp(dst[i] - max_k);
        buffer[threadIdx.x] += z;
        dst[i] = z;
    }

    __syncthreads();

    // reduce
    if (threadIdx.x == 0)
    {
        z = 0;
        for (int i = 0; i < blockDim.x; i++)
            z += buffer[i];
        buffer[SOFTMAX_THREADS] = z;
    }

    __syncthreads();

    // softmax
    Dtype sum_k = buffer[SOFTMAX_THREADS];
    for (int i = i_start; i < i_end; i += i_step)
        dst[i] /= sum_k;
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Softmax()
{
    dim3 blocks(this->rows, 1);
    dim3 threads(SOFTMAX_THREADS);
    cunn_SoftMax_updateOutput_kernel <<< blocks, threads, 0, GPUHandle::streams[streamid] >>> (this->data, this->rows, this->cols);  
}

template<typename Dtype>
__global__ void IdentityKernel(Dtype* dst, int dim) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < dim)
    {
        dst[i * dim + i] = 1.0;
    }
}

template<typename Dtype>			
void DenseMat<GPU, Dtype>::Identity(size_t dim)
{
    if (dim)
        Resize(dim, dim);
    assert(this->rows == this->cols);
    Fill(0.0);
        
    int thread_num, blocksPerGrid;
    if (this->rows < c_uCudaThreadNum)
    {
        thread_num = this->rows;
        blocksPerGrid = 1; 
    } else 
    {
        thread_num = c_uCudaThreadNum;
        blocksPerGrid = (this->rows + thread_num - 1) / thread_num;
    }
    IdentityKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid]>>>(this->data, this->cols); 
}


template<typename Dtype>
void DenseMat<GPU, Dtype>::Log(DenseMat<GPU, Dtype>& src)
{
    Resize(src.rows, src.cols);
    UnaryOp(this->data, src.data, this->count, UnaryLog<Dtype>(), streamid);    
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Log()
{
    UnaryOp(this->data, this->count, UnaryLog<Dtype>(), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Exp(DenseMat<GPU, Dtype>& src)
{
    Resize(src.rows, src.cols);
    UnaryOp(this->data, src.data, this->count, UnaryExp<Dtype>(), streamid);    
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Exp()
{
    UnaryOp(this->data, this->count, UnaryExp<Dtype>(), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Sin(DenseMat<GPU, Dtype>& src)
{
    Resize(src.rows, src.cols);
    UnaryOp(this->data, src.data, this->count, UnarySin<Dtype>(), streamid);    
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Sin()
{
    UnaryOp(this->data, this->count, UnarySin<Dtype>(), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Cos(DenseMat<GPU, Dtype>& src)
{
    Resize(src.rows, src.cols);
    UnaryOp(this->data, src.data, this->count, UnaryCos<Dtype>(), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Cos()
{
    UnaryOp(this->data, this->count, UnaryCos<Dtype>(), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Sqrt()
{
    UnaryOp(data, this->count, UnarySqrt<Dtype>(), streamid); 
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Square()
{
    UnaryOp(this->data, this->count, UnarySquare<Dtype>(), streamid); 
}

template<typename Dtype>
Dtype DenseMat<GPU, Dtype>::Norm2()
{
        cudaStreamSynchronize(GPUHandle::streams[streamid]);
	return CudaHelper_Norm2(GPUHandle::cublashandle, this->count, data); 
}

template<typename Dtype>
Dtype DenseMat<GPU, Dtype>::Asum()
{
        cudaStreamSynchronize(GPUHandle::streams[streamid]);    
	return CudaHelper_Asum(GPUHandle::cublashandle, this->count, data);
}

template<typename Dtype>
Dtype DenseMat<GPU, Dtype>::Sum()
{
        cudaStreamSynchronize(GPUHandle::streams[streamid]); 
        return thrust::reduce(dev_ptr, dev_ptr + this->count);
}


template<typename Dtype>
Dtype DenseMat<GPU, Dtype>::Dot(DenseMat<GPU, Dtype>& rhs)
{
        assert(this->count == rhs.count);
        return CudaHelper_Dot(GPUHandle::cublashandle, this->count, this->data, rhs.data);
}

template<typename Dtype>
Dtype DenseMat<GPU, Dtype>::AsScalar()
{
        cudaStreamSynchronize(GPUHandle::streams[streamid]); 
        assert(this->rows == this->cols && this->cols == 1);
        Dtype result;
        cudaMemcpy(&result, this->data, sizeof(Dtype), cudaMemcpyDeviceToHost);
        return result;
}

template<typename Dtype>
__global__ void ClipKernel(Dtype *dst, Dtype max_abs, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        if (dst[i] < -max_abs)
            dst[i] = -max_abs;
        if (dst[i] > max_abs);
            dst[i] = max_abs;
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Clip(Dtype max_abs)
{
    assert(max_abs >= 0);
    int thread_num = min(c_uCudaThreadNum, this->count);
    int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
    
    ClipKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>>(this->data, max_abs, this->count);  
}

template<typename Dtype>
Dtype DenseMat<GPU, Dtype>::Amax()
{
    cudaStreamSynchronize(GPUHandle::streams[streamid]);
    int pos;
	CudaHelper_Amax(GPUHandle::cublashandle, this->count, data, &pos);
    Dtype result;
    cudaMemcpy(&result, this->data + pos - 1, sizeof(Dtype), cudaMemcpyDeviceToHost);
    return fabs(result);
}

template<typename Dtype>
Dtype DenseMat<GPU, Dtype>::GetRowMax(size_t row_idx)
{
    assert(row_idx < this->rows);
    cudaStreamSynchronize(GPUHandle::streams[streamid]);
    size_t offset = row_idx * this->cols;
    auto iter = thrust::max_element(dev_ptr + offset, dev_ptr + offset + this->cols);
    return *iter;
}

template<typename Dtype>
size_t DenseMat<GPU, Dtype>::GetRowMaxIdx(size_t row_idx)
{
    assert(row_idx < this->rows);
    cudaStreamSynchronize(GPUHandle::streams[streamid]);
    size_t offset = row_idx * this->cols;
    auto iter = thrust::max_element(dev_ptr + offset, dev_ptr + offset + this->cols);
    return iter - dev_ptr - offset;
}

template<typename Dtype>
__global__ void MulRowKernel(Dtype *dst, Dtype* src, Dtype* factor, int cols, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < numElements)
    {    
        dst[i] += src[i] * factor[i % cols];
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::MulRowVec(DenseMat<GPU, Dtype>& src, DenseMat<GPU, Dtype>& x, Dtype beta)
{
    assert(&src != this);
	Resize(src.rows, src.cols);	
	assert(x.count == this->cols);
    Scale(beta);
    int thread_num = min(c_uCudaThreadNum, this->count);
    int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
    MulRowKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>> (this->data, src.data, x.data, this->cols, this->count); 	
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::MulColVec(DenseMat<GPU, Dtype>& src, DenseMat<GPU, Dtype>& x)
{
    assert(false);
}

template<typename Dtype>
__global__ void MulRowKernel(Dtype *dst, Dtype* factor, int cols, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < numElements)
    {    
        dst[i] = dst[i] * factor[i % cols];
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::MulRowVec(DenseMat<GPU, Dtype>& x)
{
    assert(x.count == this->cols);
    int thread_num = min(c_uCudaThreadNum, this->count);
    int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
    MulRowKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>> (this->data, x.data, this->cols, this->count);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::ReduceByRow(DenseMat<GPU, Dtype>& src, Dtype scalar)
{
    Resize(1, src.cols);
    if (bias_mult.count < src.rows)
		bias_mult.Resize(src.rows);
	bias_mult.Fill(scalar);
    Dtype alpha = 1.0, beta = 0.0;
    cudaStreamSynchronize(GPUHandle::streams[streamid]);
    CudaHelper_GeMV(GPUHandle::cublashandle, GPU_T(Trans::N), 
                    src.cols, src.rows, 
                    &alpha, src.data, src.cols,
                    bias_mult.data, 1, 
                    &beta, this->data, 1); 
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Mean(DenseMat<GPU, Dtype>& src)
{
    ReduceByRow(src, 1.0 / src.rows);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::RowSum(DenseMat<GPU, Dtype>& src)
{
    ReduceByRow(src, 1.0);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::AddRowVec(DenseMat<GPU, Dtype>& x, Dtype alpha)
{
	assert(x.count == this->cols);
		
	if (bias_mult.count < this->rows)
	{
		bias_mult.Resize(this->rows);
		bias_mult.Fill(1.0);
	}
	cudaStreamSynchronize(GPUHandle::streams[streamid]);
    // cublas is col-major
	CudaHelper_Ger(GPUHandle::cublashandle,
					this->cols, this->rows,
					&alpha,
					x.data, bias_mult.data, data);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::AddColVec(DenseMat<GPU, Dtype>& x, Dtype alpha)
{
	assert(x.count == this->rows);
	
	if (bias_mult.count < this->cols)
	{
		bias_mult.Resize(this->cols);
		bias_mult.Fill(1.0);
	}
    cudaStreamSynchronize(GPUHandle::streams[streamid]);		
    // cublas is col-major
	CudaHelper_Ger(GPUHandle::cublashandle,
					this->cols, this->rows, 
					&alpha,
					bias_mult.data, x.data, data);
}
		
template<typename Dtype>
__global__ void CSRSubmatAddKernel(Dtype* dst, int* row_ptr, int* col_idx, Dtype* val, int nnz, int n_rows, int dst_cols)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < nnz)
    {
        int l = 0, r = n_rows - 1, row;
        while (l <= r)
        {
            row = (l + r) / 2;
            if (row_ptr[row] <= i)
            {
                if (row_ptr[row + 1] > i)
                    break;
                else 
                    l = row + 1;
            } else r = row - 1;
        }
        dst[row * dst_cols + col_idx[i]] = val[i];
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::SubmatAdd(size_t row_start, size_t col_start, SparseMat<GPU, Dtype>& src, Dtype beta)
{
	assert(row_start + src.rows <= this->rows);
	assert(col_start + src.cols <= this->cols);
	Scale(beta);
    int thread_num = min(c_uCudaThreadNum, src.data->nnz);
    int blocksPerGrid = (src.data->nnz + thread_num - 1) / thread_num;
    CSRSubmatAddKernel<<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>>(this->data + row_start * this->cols + col_start, src.data->ptr, src.data->col_idx, src.data->val, src.data->nnz, src.rows, this->cols);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::SubmatAdd(size_t row_start, size_t col_start, DenseMat<GPU, Dtype>& src, Dtype beta)
{
	assert(row_start + src.rows <= this->rows);
	assert(col_start + src.cols <= this->cols);
	Scale(beta);
	
	Dtype alpha = 1.0;
	Dtype* dst_ptr = this->data + row_start * this->cols + col_start;
        cudaStreamSynchronize(GPUHandle::streams[streamid]);    
	CudaHelper_GeaM(GPUHandle::cublashandle, 
					GPU_T(Trans::N), GPU_T(Trans::N), 
					src.cols, src.rows, 
					&alpha, src.data, src.cols,
					&beta, dst_ptr, this->cols,  
					dst_ptr, this->cols);		
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::GetPointerBuf(std::vector< DenseMat<GPU, Dtype>* >& mat_list)
{
    if (mat_list.size() > pointer_buf.size())
    {
        pointer_buf.resize(mat_list.size());
    }
    for (size_t i = 0; i < mat_list.size(); ++i)
        pointer_buf[i] = mat_list[i]->data;
}

template<typename Dtype>
__global__ void ScatterColsKernel(Dtype** dst_list, Dtype* src, const int other_cols, const int this_cols, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements)
    {
        int cur_col = i % this_cols;
        Dtype* dst = dst_list[cur_col / other_cols];
        
        int dst_offset = (i / this_cols) * other_cols + cur_col % other_cols;  
        
        dst[dst_offset] = src[i];
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::ScatterCols(std::vector< DenseMat<GPU, Dtype>* >& dst_list)
{
    assert(dst_list.size() > 0);
    assert(this->cols % dst_list.size() == 0);    
    for (size_t i = 0; i < dst_list.size(); ++i)
        dst_list[i]->Resize(this->rows, this->cols / dst_list.size());
            
    GetPointerBuf(dst_list); 
    
    int thread_num = min(c_uCudaThreadNum, this->count);    
    int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
    ScatterColsKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>>(thrust::raw_pointer_cast(&pointer_buf[0]), this->data, dst_list[0]->cols, this->cols, this->count);
}

template<typename Dtype>
__global__ void GetColsFromKernel(Dtype* dst, Dtype* src, const int src_cols, const int col_start, const int dst_cols, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements) 
    {
        int cur_row = i / dst_cols;        
        int cur_col = i % dst_cols;
        dst[i] = src[col_start + cur_row * src_cols + cur_col];            
    }
}


template<typename Dtype>
void DenseMat<GPU, Dtype>::GetColsFrom(DenseMat<GPU, Dtype>& src, size_t col_start, size_t col_cnt)
{
    assert(col_start + col_cnt <= src.cols);    
    this->Resize(src.rows, col_cnt);
    
    int thread_num = min(c_uCudaThreadNum, this->count);
    int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
    GetColsFromKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>>(this->data, src.data, src.cols, col_start, col_cnt, this->count);
} 

template<typename Dtype>
__global__ void ConcatColsKernel(Dtype* dst, Dtype* src, const int src_cols, const int dst_cols, const int num_parts, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements) 
    {
        int cur_col = i % dst_cols;
        int src_offset = (i - cur_col) / num_parts; 
        dst[i] = src[src_offset + cur_col % src_cols];            
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::ConcatCols(DenseMat<GPU, Dtype>& src)
{
	assert(this->rows == src.rows);
	assert(this->cols % src.cols == 0);
    
    int num_parts = this->cols / src.cols;
    int thread_num = min(c_uCudaThreadNum, this->count);    
    int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
    ConcatColsKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>> (this->data, src.data, src.cols, this->cols, num_parts, this->count); 
}


template<typename Dtype>
__global__ void ConcatColsKernel(Dtype* dst, Dtype** src_list, const int other_cols, const int this_cols, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements) 
    {
        int cur_col = i % this_cols;
        Dtype* src = src_list[cur_col / other_cols];
        
        int src_offset = (i / this_cols) * other_cols + cur_col % other_cols;  
        dst[i] = src[src_offset];      
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::ConcatCols(std::vector< DenseMat<GPU, Dtype>* > src_list)
{
    assert(src_list.size() > 0);
    assert(src_list.size() * src_list[0]->cols == this->cols);
    
    GetPointerBuf(src_list); 
    int thread_num = min(c_uCudaThreadNum, this->count);    
    int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
    ConcatColsKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>> (this->data, thrust::raw_pointer_cast(&pointer_buf[0]), src_list[0]->cols, this->cols, this->count);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Repmat(DenseMat<GPU, Dtype>& src, size_t times_rows, size_t times_cols)
{
    assert(false);
}

template<typename Dtype>
__global__ void SparseEleWiseMulKernel(Dtype* dst, int* row_ptr, int* col_idx, Dtype* val, int n_cols, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < numElements) 
    {
        int cur_row = i / n_cols; 
        int cur_col = i % n_cols;
        
        int l = row_ptr[cur_row], r = row_ptr[cur_row + 1] - 1, idx;
        while (l <= r)
        {
            idx = (l + r) / 2;
            if (col_idx[idx] < cur_col)
                l = idx + 1;
            else if (col_idx[idx] > cur_col)
                r = idx - 1;
            else {
                dst[i] *= val[idx];
                return; 
            }
        }
        dst[i] = 0;
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::EleWiseMul(SparseMat<GPU, Dtype>& src)
{
    assert(this->rows == src.rows && this->cols == src.cols);
    int thread_num = min(c_uCudaThreadNum, this->count);    
    int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
    SparseEleWiseMulKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>> (this->data, src.data->ptr, src.data->col_idx, src.data->val, this->cols, this->count);    
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::EleWiseMul(DenseMat<GPU, Dtype>& src)
{
    assert(this->rows == src.rows && this->cols == src.cols);
    BinaryOp(this->data, src.data, this->count, BinaryMul<Dtype>(), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::EleWiseMul(DenseMat<GPU, Dtype>& lhs, DenseMat<GPU, Dtype>& rhs)
{    
    assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols);
    Resize(lhs.rows, lhs.cols);
    BinaryOp(this->data, lhs.data, rhs.data, this->count, BinaryMul<Dtype>(), streamid);            
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::EleWiseDiv(DenseMat<GPU, Dtype>& src)
{
    assert(this->rows == src.rows && this->cols == src.cols);
    BinaryOp(this->data, src.data, this->count, BinaryDiv<Dtype>(), streamid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::EleWiseDiv(DenseMat<GPU, Dtype>& lhs, DenseMat<GPU, Dtype>& rhs)
{    
    assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols);
    Resize(lhs.rows, lhs.cols);
    BinaryOp(this->data, lhs.data, rhs.data, this->count, BinaryDiv<Dtype>(), streamid);            
}
		
template<typename Dtype>
void DenseMat<GPU, Dtype>::AddSubmat(DenseMat<GPU, Dtype>& src, size_t row_start, size_t col_start, Dtype beta)
{
	assert(row_start + this->rows <= src.rows);
	assert(col_start + this->cols <= src.cols);
    
    Dtype alpha = 1.0;
	Dtype* src_ptr = src.data + row_start * src.cols + col_start;
        cudaStreamSynchronize(GPUHandle::streams[streamid]);    
	CudaHelper_GeaM(GPUHandle::cublashandle, 
					GPU_T(Trans::N), GPU_T(Trans::N), 
					this->cols, this->rows,
					&alpha, src_ptr, src.cols,
					&beta, this->data, this->cols,  
					this->data, this->cols);
}

template<typename Dtype>
__global__ void ShuffleColsKernel(Dtype* dst, Dtype* src, const int* perm, int cols, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cur_col = i % cols;
    if(i < numElements)
    {
        dst[i] = src[i + perm[cur_col] - cur_col];    
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::ShuffleCols(DenseMat<GPU, Dtype>& src, const int* perm)
{
	Resize(src.rows, src.cols);
    int thread_num = min(c_uCudaThreadNum, this->count);    
    int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
    ShuffleColsKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>> (this->data, src.data, perm, this->cols, this->count);
}

template<typename Dtype>
__global__ void ReduceColsKernel(Dtype* dst, Dtype* src, const int cols, const int num_parts, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements)
    {
        int src_offset = i % cols, j;
        src_offset += (i - src_offset) * num_parts;
        dst[i] = 0;
        for (j = 0; j < num_parts; ++j)
        {
            dst[i] += src[src_offset];
            src_offset += cols;
        }
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::ReduceCols(DenseMat<GPU, Dtype>& src)
{
	assert(src.cols % this->cols == 0);
	int num_parts = src.cols / this->cols;
    int thread_num = min(c_uCudaThreadNum, this->count);    
    int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
    ReduceColsKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>> (this->data, src.data, this->cols, num_parts, this->count);  
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::GeaM(Dtype alpha, Trans transa, DenseMat<GPU, Dtype>& A, Dtype beta, Trans transb, DenseMat<GPU, Dtype>& B)
{
	if (transa == Trans::N)
		Resize(A.rows, A.cols);
	else
		Resize(A.cols, A.rows);
		
        
	CudaHelper_GeaM(GPUHandle::cublashandle, 
					GPU_T(transa), GPU_T(transb), 
					this->cols, this->rows,  
					&alpha, A.data, A.cols, 
					&beta, B.data, B.cols, 
					data, this->cols);
}

template<typename Dtype>
__global__ void AxpbyKernel(Dtype* y, Dtype* x, Dtype a, Dtype b, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        y[i] = y[i] * b + a * x[i];
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Axpby(Dtype a, DenseMat<GPU, Dtype>& x, Dtype b)
{
    assert(x.count == this->count);
    Scale(b);
    Axpy(a, x); 
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Axpy(Dtype alpha, DenseMat<GPU, Dtype>& x)
{
	assert(x.rows == this->rows && x.cols == this->cols);
    cudaStreamSynchronize(GPUHandle::streams[streamid]);    
	CudaHelper_Axpy(GPUHandle::cublashandle, this->count, &alpha, x.data, data);	
}

template<typename Dtype>
__global__ void SpAxpyKernel(Dtype* dst, int* row_ptr, int* col_idx, Dtype* val, int nnz, int n_rows, int n_cols, Dtype alpha)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < nnz)
    {
        int l = 0, r = n_rows - 1, row;
        while (l <= r)
        {
            row = (l + r) / 2;
            if (row_ptr[row] <= i)
            {
                if (row_ptr[row + 1] > i)
                    break;
                else 
                    l = row + 1;
            } else r = row - 1;
        }
        dst[row * n_cols + col_idx[i]] += val[i] * alpha;
    }
}
    
template<typename Dtype>
void DenseMat<GPU, Dtype>::Axpy(Dtype alpha, SparseMat<GPU, Dtype>& x)
{
	assert(x.rows == this->rows && x.cols == this->cols);
	int thread_num = min(c_uCudaThreadNum, x.data->nnz);
    int blocksPerGrid = (x.data->nnz + thread_num - 1) / thread_num;
    
    SpAxpyKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>> (this->data, x.data->ptr, x.data->col_idx, x.data->val, x.data->nnz, this->rows, this->cols, alpha); 
}

template<>
void DenseMat<GPU, double>::GeMM(DenseMat<GPU, double>& A, DenseMat<GPU, double>& B, Trans transa, Trans transb, double alpha, double beta)
{
    size_t m, n, k;
	GetDims(A.rows, A.cols, transa, B.rows, B.cols, transb, m, n, k);
	Resize(m, n);
    
	cublasDgemm(GPUHandle::cublashandle,
                GPU_T(transb), GPU_T(transa), 
                n, m, k,
                &alpha, B.data, B.cols, A.data, A.cols, 
                &beta, data, this->cols);
}

template<>
void DenseMat<GPU, float>::GeMM(DenseMat<GPU, float>& A, DenseMat<GPU, float>& B, Trans transa, Trans transb, float alpha, float beta)
{
    size_t m, n, k;
	GetDims(A.rows, A.cols, transa, B.rows, B.cols, transb, m, n, k);
	Resize(m, n);
		
    cublasSgemm(GPUHandle::cublashandle,
                GPU_T(transb), GPU_T(transa), 
                n, m, k,
                &alpha, B.data, B.cols, A.data, A.cols, 
                &beta, data, this->cols);
}

template<typename Dtype>
__global__ void CSRMMKernel(Dtype alpha, int* ptr, int* col_idx, Dtype* val, Dtype* dense_data, int src_cols, Dtype* dst, int dst_cols, int numElements)
{
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset < numElements)
    {
        int i = offset / dst_cols, j = offset % dst_cols;        
        for (int t = ptr[i]; t < ptr[i + 1]; ++t)
        {
            dst[offset] += alpha * val[t] * dense_data[col_idx[t] * src_cols + j]; 
        }
    }
}


template<typename Dtype>
__global__ void CSRMMKernel_T(Dtype alpha, int n_ptr, int* ptr, int* row_idx, Dtype* val, Dtype* dense_data, int src_cols, Dtype* dst, int dst_cols)
{
    int cur_col = blockDim.x * blockIdx.x + threadIdx.x;
    if (cur_col < dst_cols)
    {
        for (int x = 0; x < n_ptr - 1; ++x)
        {
            for (int t = ptr[x]; t < ptr[x + 1]; ++t)
            {
                dst[row_idx[t] * dst_cols + cur_col] += alpha * val[t] * dense_data[x * src_cols + cur_col];
            }
        }
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::SparseMM(SparseMat<GPU, Dtype>& A, DenseMat<GPU, Dtype>& B, Trans transa, Trans transb, Dtype alpha, Dtype beta)
{
    assert(transb == Trans::N);
    size_t m, n, k;
	GetDims(A.rows, A.cols, transa, B.rows, B.cols, transb, m, n, k);
	Resize(m, n);
    this->Scale(beta);
    
    if (transa == Trans::N)
    {
        int thread_num = min(c_uCudaThreadNum, this->count);    
        int blocksPerGrid = (this->count + thread_num - 1) / thread_num;
        CSRMMKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>> (alpha, A.data->ptr, A.data->col_idx, A.data->val, B.data, B.cols, this->data, this->cols, this->count);
    } else 
    {
        int thread_num = min(c_uCudaThreadNum, this->cols);    
        int blocksPerGrid = (this->cols + thread_num - 1) / thread_num;
        CSRMMKernel_T <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[streamid] >>> (alpha, A.data->len_ptr, A.data->ptr, A.data->col_idx, A.data->val, B.data, B.cols, this->data, this->cols);
    }
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Serialize(FILE* fid)
{
        cudaStreamSynchronize(GPUHandle::streams[streamid]);    
    IMatrix<GPU, Dtype>::Serialize(fid);
	assert(fwrite(&mem_size, sizeof(size_t), 1, fid) == 1);
    Dtype* buf = new Dtype[mem_size];
    cudaMemcpy(buf, data, sizeof(Dtype) * mem_size, cudaMemcpyDeviceToHost);
	assert(fwrite(buf, sizeof(Dtype), mem_size, fid) == mem_size);
    delete[] buf;
	assert(fwrite(&is_submat, sizeof(bool), 1, fid) == 1);    
	bias_mult.Serialize(fid);
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Deserialize(FILE* fid)
{
        cudaStreamSynchronize(GPUHandle::streams[streamid]);    
	IMatrix<GPU, Dtype>::Deserialize(fid);
	assert(fread(&mem_size, sizeof(size_t), 1, fid) == 1);
	MatUtils<GPU>::DelArr(data);
	MatUtils<GPU>::MallocArr(data, sizeof(Dtype) * mem_size);
    dev_ptr = thrust::device_pointer_cast(data);
    Dtype* buf = new Dtype[mem_size];
	assert(fread(buf, sizeof(Dtype), mem_size, fid) == mem_size);
    cudaMemcpy(data, buf, sizeof(Dtype) * mem_size, cudaMemcpyHostToDevice);
    delete[] buf;
	assert(fread(&is_submat, sizeof(bool), 1, fid) == 1);
	bias_mult.Deserialize(fid);			
}

template<typename Dtype>
void DenseMat<GPU, Dtype>::Print2Screen() //debug only
{
    cudaStreamSynchronize(GPUHandle::streams[streamid]);   
	Dtype* cpu_mem = new Dtype[this->count];
	cudaMemcpy(cpu_mem, data, sizeof(Dtype) * this->count, cudaMemcpyDeviceToHost);
	std::cerr << "========mat content========" << std::endl; 			
	for (size_t i = 0; i < this->rows; ++i)
	{
		for (size_t j = 0; j < this->cols; ++j)
			std::cerr << cpu_mem[i * this->cols + j] << " ";
		std::cerr << std::endl;
	}
	std::cerr << "========    end    ========" << std::endl;
	delete[] cpu_mem;
}

template class DenseMat<GPU, double>;
template class DenseMat<GPU, float>;
