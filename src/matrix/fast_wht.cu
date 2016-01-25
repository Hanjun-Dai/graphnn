#include "fast_wht.h"
#include "fastWalshTransform_kernel_float.cuh"
#include "fastWalshTransform_kernel_double.cuh"
#include <cmath>

template<typename Dtype>
FastWHT<GPU, Dtype>::FastWHT(unsigned int _degree)
					: degree(_degree)
{
}

template<typename Dtype>
FastWHT<GPU, Dtype>::~FastWHT()
{
}

template<typename Dtype>
void FastWHT<GPU, Dtype>::Transform(size_t num_rows, Dtype* data)
{
    fwtBatchGPU(data, num_rows, degree); 
}

template class FastWHT<GPU, double>;
template class FastWHT<GPU, float>;