#include "nn/cross_entropy.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_reduce_kernel.h"
#include "tensor/cuda_helper.h"

namespace gnn
{

template<typename Dtype>
__global__ void CrossEntropyKernel(Dtype* dst, Dtype* pred_prob, int* row_ptr, int* col_idx, Dtype *val, int cols)
{
    __shared__ Dtype buffer[REDUCE_THREADS];

    Dtype* prob_row = pred_prob + blockIdx.x * cols;

    int i_start = row_ptr[blockIdx.x] + threadIdx.x;
    int i_end = row_ptr[blockIdx.x + 1];
    int i_step = blockDim.x;

    buffer[threadIdx.x] = 0;
    for (int i = i_start; i < i_end; i += i_step)
    {
    	buffer[threadIdx.x] += -cuda_log(prob_row[col_idx[i]]) * val[i];
    }
    __syncthreads();

    int shift;
    for (int i = REDUCE_THREAD_BITS - 1; i >= 0; --i)
    {
    	shift = 1 << i;
    	if (threadIdx.x < shift && threadIdx.x + shift < row_ptr[blockIdx.x + 1] - row_ptr[blockIdx.x])
    	{
    		buffer[threadIdx.x] += buffer[threadIdx.x + shift];
    	}
		__syncthreads();
    }
    if (threadIdx.x == 0)
    	dst[blockIdx.x] = buffer[0];
}

template<typename Dtype>
void CalcCrossEntropy(DTensor<GPU, Dtype>& prob, SpTensor<GPU, Dtype>& label, DTensor<GPU, Dtype>& out)
{
	ASSERT(prob.cols() == label.cols(), "# class doesn't match");
	out.Reshape({prob.rows(), 1});
	dim3 blocks(prob.rows());
	dim3 threads(REDUCE_THREADS);
    CrossEntropyKernel<<<blocks, threads, 0, cudaStreamPerThread>>> (out.data->ptr, prob.data->ptr, label.data->row_ptr, label.data->col_idx, label.data->val, label.cols());
}

template void CalcCrossEntropy(DTensor<GPU, float>& prob, SpTensor<GPU, float>& label, DTensor<GPU, float>& out);
template void CalcCrossEntropy(DTensor<GPU, double>& prob, SpTensor<GPU, double>& label, DTensor<GPU, double>& out);

}