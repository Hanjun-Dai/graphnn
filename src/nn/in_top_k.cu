#include "nn/in_top_k.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_reduce_kernel.h"
#include "tensor/cuda_helper.h"

namespace gnn
{

template<typename Dtype>
__global__ void InTopkKernel(int* dst, Dtype* pred_prob, int* label_ptr, int cols, int k)
{
    __shared__ int buffer[REDUCE_THREADS];

    Dtype* prob_row = pred_prob + blockIdx.x * cols;

    int i_start = threadIdx.x;
    int i_end = cols;
    int i_step = blockDim.x;    
    buffer[threadIdx.x] = 0;
    int target = label_ptr[blockIdx.x];
    for (int i = i_start; i < i_end; i += i_step)
    {
    	if (i != target && prob_row[i] > prob_row[target])
    		buffer[threadIdx.x]++;
    }
    __syncthreads();

    int shift;
    for (int i = REDUCE_THREAD_BITS - 1; i >= 0; --i)
    {
    	shift = 1 << i;
    	if (threadIdx.x < shift && threadIdx.x + shift < cols)
    	{
    		buffer[threadIdx.x] += buffer[threadIdx.x + shift];
    	}
		__syncthreads();
    }
    if (threadIdx.x == 0)
    	dst[blockIdx.x] = buffer[0] < k;
}

template<typename Dtype>
void IsInTopK(DTensor<GPU, Dtype>& pred, DTensor<GPU, int>& label, DTensor<GPU, int>& out, int k)
{
	ASSERT(pred.rank() == 2, "predicted prob(or logits) should be a matrix");
	ASSERT(pred.rows() == label.shape.Count(), "# instances doesn't match");
	out.Reshape(label.shape.dims);
	dim3 blocks(pred.rows());
	dim3 threads(REDUCE_THREADS);
    InTopkKernel<<<blocks, threads, 0, cudaStreamPerThread>>> (out.data->ptr, pred.data->ptr, label.data->ptr, pred.cols(), k);
}

template void IsInTopK(DTensor<GPU, float>& pred, DTensor<GPU, int>& label, DTensor<GPU, int>& out, int k);
template void IsInTopK(DTensor<GPU, double>& pred, DTensor<GPU, int>& label, DTensor<GPU, int>& out, int k);

}