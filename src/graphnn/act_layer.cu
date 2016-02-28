#include "relu_layer.h"
#include "sigmoid_layer.h"
#include "softmax_layer.h"
#include "dense_matrix.h"
#include "cuda_helper.h"
#include "cuda_unary_kernel.cuh"
#include "sparse_matrix.h"
#include <cuda_runtime.h>

#define min(x, y) (x < y ? x : y)

// =========================================== relu layer ================================================
template<typename Dtype>
void ReLULayer<GPU, Dtype>::Act(DenseMat<GPU, Dtype>& prev_out, DenseMat<GPU, Dtype>& cur_out)
{
    UnaryOp(cur_out.data, prev_out.data, prev_out.count, UnaryReLU<Dtype>(), cur_out.streamid);
}

template<typename Dtype>
__global__ void ReLUDerivKernel(Dtype *dst, Dtype *out, Dtype* cur_grad, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements && out[i] > 0)
    {
        dst[i] += cur_grad[i];
    }
}

template<typename Dtype>
void ReLULayer<GPU, Dtype>::Derivative(DenseMat<GPU, Dtype>& dst, DenseMat<GPU, Dtype>& prev_output, 
                            DenseMat<GPU, Dtype>& cur_output, DenseMat<GPU, Dtype>& cur_grad, Dtype beta)
{
    dst.Scale(beta);
    
    int thread_num = min(c_uCudaThreadNum, dst.count);    
    int blocksPerGrid = (dst.count + thread_num - 1) / thread_num;
    ReLUDerivKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[dst.streamid] >>>(dst.data, cur_output.data, cur_grad.data, dst.count);
}

template class ReLULayer<GPU, float>;
template class ReLULayer<GPU, double>;

// =========================================== sigmoid layer ================================================

template<typename Dtype>
void SigmoidLayer<GPU, Dtype>::Act(DenseMat<GPU, Dtype>& prev_out, DenseMat<GPU, Dtype>& cur_out)
{
    UnaryOp(cur_out.data, prev_out.data, prev_out.count, UnarySigmoid<Dtype>(), cur_out.streamid);
}

template<typename Dtype>
__global__ void SigmoidDerivKernel(Dtype *dst, Dtype* cur_grad, Dtype* cur_output, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        dst[i] += cur_grad[i] * cur_output[i] * (1 - cur_output[i]);
    }
}

template<typename Dtype>
void SigmoidLayer<GPU, Dtype>::Derivative(DenseMat<GPU, Dtype>& dst, DenseMat<GPU, Dtype>& prev_output, 
                            DenseMat<GPU, Dtype>& cur_output, DenseMat<GPU, Dtype>& cur_grad, Dtype beta)
{
    dst.Scale(beta);
    
    int thread_num = min(c_uCudaThreadNum, dst.count);    
    int blocksPerGrid = (dst.count + thread_num - 1) / thread_num;
    SigmoidDerivKernel <<< blocksPerGrid, thread_num, 0, GPUHandle::streams[dst.streamid] >>>(dst.data, cur_grad.data, cur_output.data, dst.count);
}

template class SigmoidLayer<GPU, float>;
template class SigmoidLayer<GPU, double>;

// =========================================== softmax layer ================================================

template<typename Dtype>
void SoftmaxLayer<GPU, Dtype>::Act(DenseMat<GPU, Dtype>& prev_out, DenseMat<GPU, Dtype>& cur_out)
{
        if (&cur_out != &prev_out)
            cur_out.CopyFrom(prev_out);
        cur_out.Softmax();
}

// Copied from https://github.com/torch/cunn/blob/master/SoftMax.cu
template<typename Dtype>
__global__ void cunn_SoftMax_updateGradInput_kernel(Dtype *gradInput, Dtype *output, Dtype *gradOutput,
                                                    int nframe, int dim)
{
  __shared__ Dtype buffer[SOFTMAX_THREADS];
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
void SoftmaxLayer<GPU, Dtype>::Derivative(DenseMat<GPU, Dtype>& dst, DenseMat<GPU, Dtype>& prev_output, 
                               DenseMat<GPU, Dtype>& cur_output, DenseMat<GPU, Dtype>& cur_grad, Dtype beta)
{
    buf.Zeros(dst.rows, dst.cols);
    dim3 blocks(cur_output.rows, 1);
    dim3 threads(SOFTMAX_THREADS);
    cunn_SoftMax_updateGradInput_kernel<<<blocks, 
                                          threads, 
                                          0, 
                                          GPUHandle::streams[buf.streamid]>>>
                                       (buf.data, cur_output.data, cur_grad.data, cur_output.rows, cur_output.cols);
    dst.Axpby(1.0, buf, beta);                                            
}

template class SoftmaxLayer<GPU, float>;
template class SoftmaxLayer<GPU, double>;