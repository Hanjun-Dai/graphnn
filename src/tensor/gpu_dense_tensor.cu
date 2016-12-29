#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "tensor/gpu_dense_tensor.h"
#include "tensor/gpu_sparse_tensor.h"
#include "tensor/cpu_dense_tensor.h"
#include "tensor/t_data.h"
#include "tensor/cu_rand_kernel.h"
#include "tensor/cuda_helper.h"
#include "tensor/gpu_unary_functor.h"
#include "tensor/gpu_binary_functor.h"
#include "tensor/gpu_reduce_kernel.h"
#include "util/mem_holder.h"

namespace gnn
{
	
template<typename Dtype>
TensorTemplate<GPU, DENSE, Dtype>::TensorTemplate() : Tensor(), data(nullptr)
{
}

template<typename Dtype>
TensorTemplate<GPU, DENSE, Dtype>::TensorTemplate(std::vector<size_t> l) : Tensor()
{
	Reshape(l);
}

template<typename Dtype>
TensorTemplate<GPU, DENSE, Dtype>::TensorTemplate(TShape s) : Tensor()
{
	Reshape(s.dims);
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<GPU, Dtype> >();

	this->data->Resize(this->shape.Count());
}

template<typename Dtype>
MatType TensorTemplate<GPU, DENSE, Dtype>::GetMatType()
{
	return MatType::dense;
}

template<typename Dtype>
MatMode TensorTemplate<GPU, DENSE, Dtype>::GetMatMode()
{
	return MatMode::gpu;
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::CopyFrom(DTensor<CPU, Dtype>& src)
{
	Reshape(src.shape.dims);
	cudaMemcpy(this->data->ptr, src.data->ptr, sizeof(Dtype) * shape.Count(), cudaMemcpyHostToDevice);
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::CopyFrom(DTensor<GPU, Dtype>& src)
{
	Reshape(src.shape.dims);
	cudaMemcpy(this->data->ptr, src.data->ptr, sizeof(Dtype) * shape.Count(), cudaMemcpyDeviceToDevice);
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::ShallowCopy(DTensor<GPU, Dtype>& src)
{
	this->shape = src.shape;
	this->data = src.data;
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Zeros()
{
	if (shape.Count())
		cudaMemset(data->ptr, 0, this->shape.Count() * sizeof(Dtype));
}

template<typename Dtype>
Dtype TensorTemplate<GPU, DENSE, Dtype>::AsScalar()
{
	ASSERT(this->shape.Count() == 1, "can only convert trivial tensor to scalar");
 	Dtype result;
 	cudaMemcpy(&result, this->data->ptr, sizeof(Dtype), cudaMemcpyDeviceToHost);
 	return result;
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::SetRandN(Dtype mean, Dtype std)
{
	SetRand(data->ptr, shape.Count(), NormalRandomizer<Dtype>(mean, std));
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::SetRandU(Dtype lb, Dtype ub)
{
	SetRand(data->ptr, shape.Count(), UniformRandomizer<Dtype>(lb, ub));
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Fill(Dtype scalar)
{
	if (scalar == 0)
		this->Zeros();
	else {
		UnaryEngine<GPU>::Exec<UnarySet>(this->data->ptr, this->shape.Count(), scalar);
	}
}

template<typename Dtype>
Dtype TensorTemplate<GPU, DENSE, Dtype>::ASum()
{
	Dtype result;
	WITH_GPUCTX(ctx, {
		result = Cuda_Asum(ctx.cublasHandle, shape.Count(), data->ptr);
	});
	return result;
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::ArgMax(DTensor<GPU, int>& dst, uint axis)
{
	ASSERT(axis == 0, "not supported for axis > 0 in GPU DENSE Tensor");
	dst.Reshape({this->shape[0]});
	MatColReduce::Exec(dst.data->ptr, this->data->ptr, this->shape[0], this->shape.Count(1), MaxIdxReduce<Dtype>());
}

template<>
void TensorTemplate<GPU, DENSE, double>::MM(DTensor<GPU, double>& a, DTensor<GPU, double>& b, Trans transA, Trans transB, double alpha, double beta)
{
	ASSERT(a.rank() == 2 && b.rank() == 2, "only support mat x mat now");
	size_t m, n, k;
	GetDims(a.rows(), a.cols(), transA, b.rows(), b.cols(), transB, m, n, k);
	
	Reshape({m, n});
	WITH_GPUCTX(ctx, {
		cublasDgemm(ctx.cublasHandle, 
	                GPU_T(transB), GPU_T(transA), 
	                n, m, k,
	                &alpha, b.data->ptr, b.cols(), a.data->ptr, a.cols(), 
	                &beta, data->ptr, this->cols());
	});	
}

template<>
void TensorTemplate<GPU, DENSE, float>::MM(DTensor<GPU, float>& a, DTensor<GPU, float>& b, Trans transA, Trans transB, float alpha, float beta)
{
	ASSERT(a.rank() == 2 && b.rank() == 2, "only support mat x mat now");
	size_t m, n, k;
	GetDims(a.rows(), a.cols(), transA, b.rows(), b.cols(), transB, m, n, k);
	
	Reshape({m, n});
	WITH_GPUCTX(ctx, {
	    cublasSgemm(ctx.cublasHandle,
	                GPU_T(transB), GPU_T(transA), 
	                n, m, k,
	                &alpha, b.data->ptr, b.cols(), a.data->ptr, a.cols(), 
	                &beta, data->ptr, this->cols());
	});	
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
void TensorTemplate<GPU, DENSE, Dtype>::MM(SpTensor<GPU, Dtype>& a, DTensor<GPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{
    assert(transB == Trans::N);
	size_t m, n, k;
	GetDims(a.rows(), a.cols(), transA, b.rows(), b.cols(), transB, m, n, k);

	Reshape({m, n});
    this->Scale(beta);
    if (transA == Trans::N)
    {
        int thread_num = min(c_uCudaThreadNum, this->shape.Count());
        int blocksPerGrid = (this->shape.Count() + thread_num - 1) / thread_num;
        CSRMMKernel <<< blocksPerGrid, thread_num, 0, cudaStreamPerThread >>> (alpha, a.data->row_ptr, a.data->col_idx, a.data->val, b.data->ptr, b.cols(), this->data->ptr, this->cols(), this->shape.Count());
    } else 
    {
        DTensor<GPU, Dtype> bt(b.shape);
        DTensor<GPU, Dtype> c({m, n});
        WITH_GPUCTX(ctx, {
            Dtype one = 1.0;
            Dtype zero = 0.0;
            Cuda_GeaM(ctx.cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_T, 
                    b.rows(), b.cols(), &one, b.data->ptr, b.cols(), &zero, b.data->ptr, b.cols(), bt.data->ptr, b.rows());
                               
            Cuda_CSRMM(ctx.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, 
                    a.rows(), b.cols(), a.cols(), a.data->nnz, &alpha, 
                    a.data->val, a.data->row_ptr, a.data->col_idx, bt.data->ptr, bt.rows(), &beta, c.data->ptr, c.rows());                
            Cuda_GeaM(ctx.cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_T, 
                    cols(), rows(), &one, c.data->ptr, c.rows(), &zero, c.data->ptr, c.rows(), data->ptr, n);
        });
    }
}

// Copied from https://github.com/torch/cunn/blob/master/SoftMax.cu
template<typename Dtype>
__global__ void cunn_SoftMax_updateOutput_kernel(Dtype *orig_ptr, int batch_size, int dim)
{
    __shared__ Dtype buffer[REDUCE_THREADS + 1];
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
        buffer[REDUCE_THREADS] = z;
    }

    __syncthreads();

    // sum?
    Dtype max_k = buffer[REDUCE_THREADS];
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
        buffer[REDUCE_THREADS] = z;
    }

    __syncthreads();

    // softmax
    Dtype sum_k = buffer[REDUCE_THREADS];
    for (int i = i_start; i < i_end; i += i_step)
        dst[i] /= sum_k;
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Softmax()
{
	ASSERT(this->rank() == 2, "Softmax is assumed to exec on matrix");
    dim3 blocks(this->rows(), 1);
    dim3 threads(REDUCE_THREADS);
    cunn_SoftMax_updateOutput_kernel <<< blocks, threads, 0, cudaStreamPerThread >>> (this->data->ptr, this->rows(), this->cols());  
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Scale(Dtype scalar)
{
	if (scalar == 0)
	{
		Zeros();
		return;
	}	
	if (scalar != 1)
	{
		UnaryEngine<GPU>::Exec<UnaryScale>(this->data->ptr, this->shape.Count(), scalar);
	}
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Mean(DTensor<GPU, Dtype>& a, int axis)
{
	ASSERT(axis == -1, "currently only support global mean");
	Reshape({1});
	MatColReduce::Exec(this->data->ptr, a.data->ptr, 1, a.shape.Count(), SumReduce<Dtype>());
	Scale(1.0 / a.shape.Count());
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Add(Dtype scalar)
{
	UnaryEngine<GPU>::Exec<UnaryAdd>(this->data->ptr, this->shape.Count(), scalar);
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Axpy(Dtype a, DTensor<GPU, Dtype>& x)
{
	ASSERT(this->shape == x.shape, "shape doesn't match in Axpy");
	WITH_GPUCTX(ctx, {
		Cuda_Axpy(ctx.cublasHandle, this->shape.Count(), &a, x.data->ptr, data->ptr);
	});
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
void TensorTemplate<GPU, DENSE, Dtype>::Axpy(Dtype a, SpTensor<GPU, Dtype>& x)
{
	ASSERT(this->shape == x.shape, "shape doesn't match in Axpy");
	int thread_num = min(c_uCudaThreadNum, x.data->nnz);
    int blocksPerGrid = (x.data->nnz + thread_num - 1) / thread_num;

    SpAxpyKernel <<< blocksPerGrid, thread_num, 0, cudaStreamPerThread >>> (this->data->ptr, x.data->row_ptr, x.data->col_idx, x.data->val, x.data->nnz, this->rows(), this->cols(), a); 
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Axpby(Dtype a, DTensor<GPU, Dtype>& x, Dtype b)
{
	ASSERT(this->shape == x.shape, "shape doesn't match in Axpby");
    Scale(b);
    Axpy(a, x); 
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
void TensorTemplate<GPU, DENSE, Dtype>::ElewiseMul(SpTensor<GPU, Dtype>& src)
{
	ASSERT(this->shape == src.shape, "shape doesn't match in ElewiseMul");
	int thread_num = min(c_uCudaThreadNum, this->shape.Count());
    int blocksPerGrid = (this->shape.Count() + thread_num - 1) / thread_num;
    SparseEleWiseMulKernel <<< blocksPerGrid, thread_num, 0, cudaStreamPerThread >>> (this->data->ptr, src.data->row_ptr, src.data->col_idx, src.data->val, this->cols(), this->shape.Count());
}

template<typename Dtype>
__global__ void BCastEleWiseMulKernel(Dtype* dst, size_t* dst_shape, Dtype* src, size_t* src_shape, int rank, size_t* offset, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
	{
		int src_idx = 0, d_off = i;
		for (int t = rank - 1; t >= 0; --t)
		{
			int cur_coor = d_off % dst_shape[t];
			if (cur_coor < src_shape[t])
				src_idx += cur_coor * offset[t];
			d_off /= dst_shape[t];
		}
		dst[i] *= src[src_idx];
	}
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::ElewiseMul(DTensor<GPU, Dtype>& src)
{
	if (this->shape == src.shape)
	{
		BinaryEngine<GPU>::Exec<BinaryMul>(this->data->ptr, src.data->ptr, this->shape.Count());
	} else { // require broadcasting
		ASSERT(this->rank() == src.rank(), "broadcasting only support same rank tensors; please do reshape manually");
		for (size_t i = 0; i < this->rank(); ++i)
			if (shape.dims[i] != src.shape.dims[i])
				ASSERT(src.shape.dims[i] == 1, "shape mismatch, broadcasting failed");
		int thread_num = min(c_uCudaThreadNum, this->shape.Count());
    	int blocksPerGrid = (this->shape.Count() + thread_num - 1) / thread_num;

    	std::vector<size_t> offset(rank());
    	for (size_t i = 0; i + 1 < rank(); ++i)
    		offset[i] = src.shape.Count(i + 1);
    	offset[offset.size() - 1] = 1;

    	thrust::device_vector<size_t> src_shape(src.shape.dims.begin(), src.shape.dims.end());
    	thrust::device_vector<size_t> dst_shape(shape.dims.begin(), shape.dims.end());
    	thrust::device_vector<size_t> dev_off(offset.begin(), offset.end());

    	size_t* ss = thrust::raw_pointer_cast(&src_shape[0]);
    	size_t* ds = thrust::raw_pointer_cast(&dst_shape[0]);
    	size_t* p_off = thrust::raw_pointer_cast(&dev_off[0]);

    	BCastEleWiseMulKernel <<< blocksPerGrid, thread_num, 0, cudaStreamPerThread >>>(data->ptr, ds, src.data->ptr, ss, rank(), p_off, shape.Count());
	}
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Inv()
{
	UnaryEngine<GPU>::Exec<UnaryInv>(this->data->ptr, this->shape.Count());
}

template<typename Dtype>
Dtype TensorTemplate<GPU, DENSE, Dtype>::Norm2()
{
	Dtype result;
	WITH_GPUCTX(ctx, {
		result = Cuda_Norm2(ctx.cublasHandle, this->shape.Count(), data->ptr);
	});
	return result;
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Square()
{
	UnaryEngine<GPU>::Exec<UnarySquare>(this->data->ptr, this->shape.Count());
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Sqrt()
{
	UnaryEngine<GPU>::Exec<UnarySqrt>(this->data->ptr, this->shape.Count());
}

template class TensorTemplate<GPU, DENSE, float>;
template class TensorTemplate<GPU, DENSE, double>;

///================================ int tensor ===================================

TensorTemplate<GPU, DENSE, int>::TensorTemplate() : data(nullptr)
{

}

void TensorTemplate<GPU, DENSE, int>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<GPU, int> >();

    this->data->Resize(this->shape.Count());
}

MatType TensorTemplate<GPU, DENSE, int>::GetMatType()
{
	return MatType::dense;
}

MatMode TensorTemplate<GPU, DENSE, int>::GetMatMode()
{
	return MatMode::gpu;
}

void TensorTemplate<GPU, DENSE, int>::CopyFrom(DTensor<CPU, int>& src)
{
    Reshape(src.shape.dims);
    cudaMemcpy(this->data->ptr, src.data->ptr, sizeof(int) * shape.Count(), cudaMemcpyHostToDevice);
}

void TensorTemplate<GPU, DENSE, int>::CopyFrom(DTensor<GPU, int>& src)
{
    Reshape(src.shape.dims);
    cudaMemcpy(this->data->ptr, src.data->ptr, sizeof(int) * shape.Count(), cudaMemcpyDeviceToDevice);
}

void TensorTemplate<GPU, DENSE, int>::ShallowCopy(DTensor<GPU, int>& src)
{
    this->shape = src.shape;
    this->data = src.data;
}

void TensorTemplate<GPU, DENSE, int>::Zeros()
{
    if (shape.Count())
        cudaMemset(data->ptr, 0, this->shape.Count() * sizeof(int));
}

void TensorTemplate<GPU, DENSE, int>::Fill(int scalar)
{
    if (scalar == 0)
        this->Zeros();
    else {
        UnaryEngine<GPU>::Exec<UnarySet>(this->data->ptr, this->shape.Count(), scalar);
    }
}

int TensorTemplate<GPU, DENSE, int>::AsScalar()
{
    ASSERT(this->shape.Count() == 1, "can only convert trivial tensor to scalar");
    int result;
    cudaMemcpy(&result, this->data->ptr, sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}

template class TensorTemplate<GPU, DENSE, int>;

}