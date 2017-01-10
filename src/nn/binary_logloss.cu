#include "nn/binary_logloss.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_reduce_kernel.h"
#include "tensor/cuda_helper.h"

namespace gnn
{

template<typename Dtype>
void CalcBinaryLogLoss(DTensor<GPU, Dtype>& prob, DTensor<GPU, Dtype>& label, DTensor<GPU, Dtype>& out)
{
	ASSERT(false, "no impl");
}

template void CalcBinaryLogLoss(DTensor<GPU, float>& prob, DTensor<GPU, float>& label, DTensor<GPU, float>& out);
template void CalcBinaryLogLoss(DTensor<GPU, double>& prob, DTensor<GPU, double>& label, DTensor<GPU, double>& out);

}