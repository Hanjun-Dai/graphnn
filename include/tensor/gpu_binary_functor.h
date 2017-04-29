#ifndef GPU_BINARY_FUNCTOR_H
#define GPU_BINARY_FUNCTOR_H

#ifdef USE_GPU

#include "binary_functor.h"
#include "gpu_handle.h"

#define min(x, y) (x < y ? x : y)

namespace gnn
{

/**
 * @brief      Class for binary mul. GPU specialization
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class BinaryMul<GPU, Dtype>
{
public:
	/**
	 * @brief      constructor
	 */
	BinaryMul() {}

	__device__ inline void operator()(Dtype& dst, const Dtype& lhs)
	{
		dst *= lhs;
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& lhs, const Dtype& rhs)
	{
		dst = lhs * rhs;
	}
};


template<typename Dtype, class BinaryEngine>
__global__ void BinaryKernel(Dtype *dst, const Dtype *lhs, int numElements, BinaryEngine binary)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < numElements)
    {
        binary(dst[i], lhs[i]);
    }
}

/**
 * @brief      Class for binary engine, GPU specialization
 */
template<>
class BinaryEngine<GPU>
{
public:
	/**
	 * @brief      Execute the binary operation
	 *
	 * @param      dst        The destination ptr
	 * @param      src        The source ptr
	 * @param[in]  count      # elements to operate
	 * @param[in]  <unnamed>  { other parameter required by specific functor }
	 *
	 * @tparam     Functor    { Functor class type }
	 * @tparam     Dtype      { float/double/int }
	 * @tparam     Args       { extra arguements required by specific functor }
	 */
	template<template <typename, typename> class Functor, typename Dtype, typename... Args>
	static void Exec(Dtype* dst, Dtype* src, size_t count, Args&&... args)
	{
		Functor<GPU, Dtype> func(std::forward<Args>(args)...);
		Exec(dst, src, count, func);
	}

	/**
	 * @brief      Execute the binary operation
	 *
	 * @param      dst      The destination
	 * @param      src      The source
	 * @param[in]  count    # elements to operate
	 * @param[in]  f        the functor
	 *
	 * @tparam     Dtype    { float/double/in }
	 * @tparam     Functor  { the functor class }
	 */
	template<typename Dtype, typename Functor>
	static void Exec(Dtype* dst, Dtype* src, size_t count, Functor f)
	{
    	int thread_num = min(c_uCudaThreadNum, count);    
    	int blocksPerGrid = (count + thread_num - 1) / thread_num;
    	BinaryKernel<<<blocksPerGrid, thread_num, 0, cudaStreamPerThread>>> (dst, src, count, f);
	}
};

}

#endif

#endif