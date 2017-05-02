#ifndef GPU_UNARY_FUNCTOR_H
#define GPU_UNARY_FUNCTOR_H

#ifdef USE_GPU
#include "unary_functor.h"
#include "cuda_helper.h"
#define min(x, y) (x < y ? x : y)

namespace gnn
{

/**
 * @brief      GPU specialization of UnarySet
 *
 * @tparam     Dtype  { float/double/int }
 */
template<typename Dtype>
class UnarySet<GPU, Dtype>
{
public:
	/**
	 * @brief      constructor
	 *
	 * @param[in]  _scalar  The scalar to be set
	 */
	UnarySet(Dtype _scalar) : scalar(_scalar) {}

	/**
	 * set scalar to dst
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst = scalar;
	}

private:
	/**
	 * scalar to be set
	 */
	Dtype scalar;
};

/**
 * @brief      GPU specialization of UnaryTruncate
 *
 * @tparam     Dtype  { float/double/int }
 */
template<typename Dtype>
class UnaryTruncate<GPU, Dtype>
{
public:
	UnaryTruncate(Dtype lb, Dtype ub) : lower_bound(lb), upper_bound(ub) {}

	/**
	 * truncate dst
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst = dst < lower_bound ? lower_bound : dst;
		dst = dst > upper_bound ? upper_bound : dst;
	}

private:
	/**
	 * lower bound
	 */
	Dtype lower_bound;
	/**
	 * upper bound
	 */
	Dtype upper_bound;
};

/**
 * @brief      GPU specialization of UnaryScale
 *
 * @tparam     Dtype  { float/double/int }
 */
template<typename Dtype>
class UnaryScale<GPU, Dtype>
{
public:
	/**
	 * @brief      constructor
	 *
	 * @param[in]  _scalar  The scalar to multiply
	 */
	UnaryScale(Dtype _scalar) : scalar(_scalar) {}

	/**
	 * scale dst by scalar
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst *= scalar;
	}

private:
	/**
	 * scalar to multiply
	 */
	Dtype scalar;
};

/**
 * @brief      GPU specialization of UnaryAdd
 *
 * @tparam     Dtype  { float/double/int }
 */
template<typename Dtype>
class UnaryAdd<GPU, Dtype>
{
public:
	/**
	 * @brief      constructor
	 *
	 * @param[in]  _scalar  The scalar to add
	 */
	UnaryAdd(Dtype _scalar) : scalar(_scalar) {}

	/**
	 * add dst by scalar
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst += scalar;
	}

private:
	/**
	 * scalar to add
	 */
	Dtype scalar;
};

/**
 * @brief      UnaryAbs
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnaryAbs<GPU, Dtype>
{
public:
	/**
	 * abs dst
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst = cuda_fabs(dst);
	}
};

/**
 * @brief      UnaryInv
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnaryInv<GPU, Dtype>
{
public:
	/**
	 * inverse dst
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst = 1.0 / dst;
	}
};


/**
 * @brief      UnaryReLU
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnaryReLU<GPU, Dtype>
{
public:
	/**
	 * rectify dst
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		if (dst < 0)
			dst = 0;
	}
};

/**
 * @brief      UnarySigmoid
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnarySigmoid<GPU, Dtype>
{
public:
	/**
	 * dst = 1 / (1 + exp(-dst))
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst = 1.0 / (1.0 + cuda_exp(-dst));
	}
};

/**
 * @brief      UnaryTanh
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnaryTanh<GPU, Dtype>
{
public:
	/**
	 * dst = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
	 */
	__device__ inline void operator()(Dtype& dst)
	{
        Dtype x = cuda_exp(dst);
        Dtype y = cuda_exp(-dst);
        dst = (x - y) / (x + y);
	}
};


/**
 * @brief      UnaryLog
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnaryLog<GPU, Dtype>
{
public:
	/**
	 * dst = log(dst)
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst = cuda_log(dst);
	}
};

/**
 * @brief      UnaryInvSqrt
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnaryInvSqrt<GPU, Dtype>
{
public:
	/**
	 * dst = 1.0 / sqrt(dst)
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst = cuda_rsqrt(dst);
	}
};

/**
 * @brief      UnaryExp
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnaryExp<GPU, Dtype>
{
public:
	/**
	 * dst = exp(dst)
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst = cuda_exp(dst);
	}
};

/**
 * @brief      UnarySquare
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class UnarySquare<GPU, Dtype>
{
public:
	/**
	 * inverse dst
	 */
	__device__ inline void operator()(Dtype& dst)
	{
		dst = dst * dst;
	}
};

/**
 * @brief      Class for unary sqrt. GPU float specialization
 */
template<>
class UnarySqrt<GPU, float>
{
public:
	__device__ inline void operator()(float& dst)
	{
		dst = sqrtf(dst);
	}	
};

/**
 * @brief      Class for unary sqrt. GPU double specialization
 */
template<>
class UnarySqrt<GPU, double>
{
public:
	__device__ inline void operator()(double& dst)
	{
		dst = sqrt(dst);
	}	
};

//=================================== GPU call interface ======================================

template<typename Dtype, class UnaryEngine>
__global__ void UnaryKernel(Dtype *dst, int numElements, UnaryEngine unary)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < numElements)
    {
        unary(dst[i]);
    }
}

/**
 * @brief      Class for unary engine, GPU specialization
 */
template<>
class UnaryEngine<GPU>
{
public:
	/**
	 * @brief      Execute the unary operation
	 *
	 * @param      data       The data pointer
	 * @param[in]  count      # elements to operate
	 * @param[in]  <unnamed>  { other parameter required by specific functor }
	 *
	 * @tparam     Functor    { Functor class type }
	 * @tparam     Dtype      { float/double/int }
	 * @tparam     Args       { extra arguements required by specific functor }
	 */
	template<template <typename, typename> class Functor, typename Dtype, typename... Args>
	static void Exec(Dtype* data, size_t count, Args&&... args)
	{
		Functor<GPU, Dtype> func(std::forward<Args>(args)...);
		Exec(data, count, func);
	}

	/**
	 * @brief      Execute the unary operation
	 *
	 * @param      data     The data pointer 
	 * @param[in]  count    # elements to operate
	 * @param[in]  f        { the functor object }
	 *
	 * @tparam     Dtype    { float/double/int }
	 * @tparam     Functor  { The functor class }
	 */
	template<typename Dtype, typename Functor>
	static void Exec(Dtype* data, size_t count, Functor f)
	{
    	int thread_num = min(c_uCudaThreadNum, count);    
    	int blocksPerGrid = (count + thread_num - 1) / thread_num;
    	UnaryKernel<<<blocksPerGrid, thread_num, 0, cudaStreamPerThread>>> (data, count, f);
	}
};

}
#endif

#endif