#include "nn/entropy.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_unary_functor.h"

namespace gnn
{

template<typename Dtype>
__global__ void CalcEntropyKernel(Dtype *prob, Dtype *out, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        auto p = prob[i];
        out[i] = -p * log(p) - (1 - p) * log(1 - p);
    }
}

template<typename Dtype>
void CalcEntropy(DTensor<GPU, Dtype>& prob, DTensor<GPU, Dtype>& out)
{
    ASSERT(prob.cols() == 1, "only support binary now");
    out.Reshape({prob.rows(), 1});

    int thread_num = c_uCudaThreadNum;
    if (prob.shape.Count() < thread_num)
        thread_num = prob.shape.Count();
    int blocksPerGrid = (prob.shape.Count() + thread_num - 1) / thread_num;
    CalcEntropyKernel <<< blocksPerGrid, thread_num, 0, cudaStreamPerThread >>>(prob.data->ptr, out.data->ptr, prob.shape.Count());
}

template void CalcEntropy(DTensor<GPU, float>& prob, DTensor<GPU, float>& out);
template void CalcEntropy(DTensor<GPU, double>& prob, DTensor<GPU, double>& out);


}