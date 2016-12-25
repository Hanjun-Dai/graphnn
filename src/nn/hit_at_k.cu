#include "nn/hit_at_k.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_reduce_kernel.h"
#include "tensor/cuda_helper.h"

namespace gnn
{
template<typename Dtype>
__global__ void HitsTopkKernel(int* dst, Dtype* pred_prob, int* row_ptr, int* col_idx, Dtype *val, int cols, int k)
{
    __shared__ int buffer[REDUCE_THREADS];

    Dtype* prob_row = pred_prob + blockIdx.x * cols;

    int i_start = threadIdx.x;
    int i_end = cols;
    int i_step = blockDim.x;
    if (i_start < cols)
    	buffer[threadIdx.x] = i_start;
    for (int i = i_start + i_step; i < i_end; i += i_step)
    {
    	if (prob_row[i] > prob_row[buffer[threadIdx.x]])
    		buffer[threadIdx.x] = i;
    }
    __syncthreads();

    int shift;
    for (int i = REDUCE_THREAD_BITS - 1; i >= 0; --i)
    {
    	shift = 1 << i;
    	if (threadIdx.x < shift && threadIdx.x + shift < cols)
    	{
    		if (prob_row[buffer[threadIdx.x + shift]] > prob_row[buffer[threadIdx.x]])
    			buffer[threadIdx.x] = buffer[threadIdx.x + shift];
    	}
		__syncthreads();
    }
    if (threadIdx.x == 0)
    	dst[blockIdx.x] = buffer[0];
	__syncthreads(); // find top 1

   	i_start = row_ptr[blockIdx.x] + threadIdx.x;
    i_end = row_ptr[blockIdx.x + 1];
    i_step = blockDim.x;
    buffer[threadIdx.x] = 0;
    for (int i = i_start; i < i_end; i += i_step)
    {
    	if (col_idx[i] == dst[blockIdx.x])
    		buffer[threadIdx.x] = 1;
    }
    __syncthreads();

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
void HitsTopK(DTensor<GPU, Dtype>& pred, SpTensor<GPU, Dtype>& label, DTensor<GPU, int>& out, int k)
{
	ASSERT(pred.rank() == 2, "predicted prob(or logits) should be a matrix");
	ASSERT(pred.rows() == label.rows(), "# instances doesn't match");	
	ASSERT(k == 1, "gpu only support k=1 now");
	out.Reshape({pred.rows(), 1});	

	dim3 blocks(pred.rows());
	dim3 threads(REDUCE_THREADS);
    HitsTopkKernel<<<blocks, threads, 0, cudaStreamPerThread>>> (out.data->ptr, pred.data->ptr, label.data->row_ptr, label.data->col_idx, label.data->val, pred.cols(), k);
}

template void HitsTopK(DTensor<GPU, float>& pred, SpTensor<GPU, float>& label, DTensor<GPU, int>& out, int k);
template void HitsTopK(DTensor<GPU, double>& pred, SpTensor<GPU, double>& label, DTensor<GPU, int>& out, int k);

}