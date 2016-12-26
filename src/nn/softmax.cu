#include "nn/relu.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_unary_functor.h"
#include "tensor/gpu_reduce_kernel.h"

namespace gnn
{

// Copied from https://github.com/torch/cunn/blob/master/SoftMax.cu
template<typename Dtype>
__global__ void cunn_SoftMax_updateGradInput_kernel(Dtype *gradInput, Dtype *output, Dtype *gradOutput,
                                                    int nframe, int dim)
{
  __shared__ Dtype buffer[REDUCE_THREADS];
  Dtype *gradInput_k = gradInput + blockIdx.x*dim + blockIdx.y;
  Dtype *output_k = output + blockIdx.x*dim + blockIdx.y;
  Dtype *gradOutput_k = gradOutput + blockIdx.x*dim + blockIdx.y;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // sum?
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[threadIdx.x] += gradOutput_k[i] * output_k[i];

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    Dtype sum_k = 0;
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[0] = sum_k;
  }

  __syncthreads();

  Dtype sum_k = buffer[0];
  for (int i=i_start; i<i_end; i+=i_step)
    gradInput_k[i] = output_k[i] * (gradOutput_k[i] - sum_k);
}

template<typename Dtype>
void SoftmaxDeriv(DTensor<GPU, Dtype>& dst, DTensor<GPU, Dtype>& cur_output, DTensor<GPU, Dtype>& cur_grad)
{
	DTensor<GPU, Dtype> buf(dst.shape);
	buf.Zeros();
	dim3 blocks(cur_output.rows(), 1);
    dim3 threads(REDUCE_THREADS);
    cunn_SoftMax_updateGradInput_kernel<<<blocks, 
                                          threads, 
                                          0, 
                                          cudaStreamPerThread>>>
                                       (buf.data->ptr, cur_output.data->ptr, cur_grad.data->ptr, cur_output.rows(), cur_output.cols());
    dst.Axpy(1.0, buf);
}

template void SoftmaxDeriv(DTensor<GPU, float>& dst, DTensor<GPU, float>& cur_output, DTensor<GPU, float>& cur_grad);
template void SoftmaxDeriv(DTensor<GPU, double>& dst, DTensor<GPU, double>& cur_output, DTensor<GPU, double>& cur_grad);


}