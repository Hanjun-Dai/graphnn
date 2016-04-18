#ifndef CUDA_UNARY_KERNEL_CUH
#define CUDA_UNARY_KERNEL_CUH

#include <cuda_runtime.h>
#include "gpuhandle.h"
#include "cuda_helper.h"

//=================================== power ======================================

template<typename Dtype>
class UnaryPow
{
public:
	UnaryPow(Dtype _scalar) : scalar(_scalar) {}
    
	__device__ inline void operator()(Dtype& dst)
	{
		dst = cuda_pow(dst, scalar);
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = cuda_pow(src, scalar);
	}
    
private:
    Dtype scalar;    
};

//=================================== scale ======================================

template<typename Dtype>
class UnaryScale
{
public:
	UnaryScale(Dtype _scalar) : scalar(_scalar) {}
    
	__device__ inline void operator()(Dtype& dst)
	{
		dst *= scalar;
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = src * scalar;
	}
    
private:
    Dtype scalar;    
};

//=================================== sqrt ======================================

template<typename Dtype>
class UnarySqrt{};

template<>
class UnarySqrt<float>
{
public:
    UnarySqrt() {}
    __device__ inline void operator()(float& dst)
	{
		dst = sqrtf(dst);
	}
    
    __device__ inline void operator()(float& dst, const float& src)
	{
		dst = sqrtf(src);
	}
};

template<>
class UnarySqrt<double>
{
public:
    UnarySqrt() {}
    __device__ inline void operator()(double& dst)
	{
		dst = sqrt(dst);
	}
    
    __device__ inline void operator()(double& dst, const double& src)
	{
		dst = sqrt(src);
	}
};

//=================================== set ======================================

template<typename Dtype>
class UnarySet
{
public:
    UnarySet(Dtype _scalar) : scalar(_scalar) {}
    
    __device__ inline void operator()(Dtype& dst)
	{
		dst = scalar;
	}
    
private:
    Dtype scalar;    
};

//=================================== add ======================================

template<typename Dtype>
class UnaryAdd
{
public:
    UnaryAdd(Dtype _scalar) : scalar(_scalar) {}
    
    __device__ inline void operator()(Dtype& dst)
	{
		dst += scalar;
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = src + scalar;
	}
    
private:
    Dtype scalar;    
};

//=================================== inv ======================================

template<typename Dtype>
class UnaryInv
{
public:
    UnaryInv() {}
    
    __device__ inline void operator()(Dtype& dst)
	{
		dst = 1.0 / dst;
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = 1.0 / src;
	}
};

//=================================== inv_sqrt ======================================

template<typename Dtype>
class UnaryInvSqrt
{
public:
    UnaryInvSqrt() {}
    
    __device__ inline void operator()(Dtype& dst)
	{
		dst = my_inv_sqrt(dst);
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = my_inv_sqrt(src);
	}
    
private:
    __device__ inline float my_inv_sqrt(const float& src)
    {
        return rsqrtf(src);
    }
    
    __device__ inline double my_inv_sqrt(const double& src)
    {
        return rsqrt(src);
    }
};

//=================================== sin ======================================

template<typename Dtype>
class UnarySin
{
public:
    UnarySin() {}
    
    __device__ inline void operator()(Dtype& dst)
	{
		dst = my_sin(dst);
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = my_sin(src);
	}
    
private:
    __device__ inline float my_sin(const float& src)
    {
        return sinf(src);
    }
    
    __device__ inline double my_sin(const double& src)
    {
        return sin(src);
    }
};

//=================================== exp ======================================

template<typename Dtype>
class UnaryExp
{
public:
    UnaryExp() {}
    
    __device__ inline void operator()(Dtype& dst)
	{
		dst = cuda_exp(dst);
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = cuda_exp(src);
	}
};

//=================================== log ======================================
template<typename Dtype>
class UnaryLog
{
public:
    UnaryLog() {}
    
    __device__ inline void operator()(Dtype& dst)
	{
		dst = cuda_log(dst);
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = cuda_log(src);
	}
};

//=================================== sigmoid ======================================

template<typename Dtype>
class UnarySigmoid
{
public:
    UnarySigmoid() {}
    
    __device__ inline void operator()(Dtype& dst)
	{
		dst = 1.0 / (1.0 + cuda_exp(-dst));
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = 1.0 / (1.0 + cuda_exp(-src));
	}
};

//=================================== cos ======================================

template<typename Dtype>
class UnaryCos
{
public:
    UnaryCos() {}
    
    __device__ inline void operator()(Dtype& dst)
	{
		dst = my_cos(dst);
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = my_cos(src);
	}
    
private:
    __device__ inline float my_cos(const float& src)
    {
        return cosf(src);
    }
    
    __device__ inline double my_cos(const double& src)
    {
        return cos(src);
    }
};

//=================================== square ======================================

template<typename Dtype>
class UnarySquare
{
public:
	UnarySquare() {}
    
	__device__ inline void operator()(Dtype& dst)
	{
		dst = dst * dst;
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = src * src;
	}    
};

//=================================== relu ======================================

template<typename Dtype>
class UnaryReLU
{
public:
	UnaryReLU() {}
    
	__device__ inline void operator()(Dtype& dst)
	{
		dst = dst > 0 ? dst : 0;
	}
    
    __device__ inline void operator()(Dtype& dst, const Dtype& src)
	{
		dst = src > 0 ? src : 0;
	}
};


//=================================== call interface ======================================

template<typename Dtype, class UnaryEngine>
__global__ void UnaryKernel(Dtype *dst, int numElements, UnaryEngine unary)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < numElements)
    {
        unary(dst[i]);
    }
}

template<typename Dtype, class UnaryEngine>
void UnaryOp(Dtype *dst, int numElements, UnaryEngine unary, const unsigned& sid)
{
    int thread_num = min(c_uCudaThreadNum, numElements);    
    int blocksPerGrid = (numElements + thread_num - 1) / thread_num;
    UnaryKernel<<<blocksPerGrid, thread_num, 0, GPUHandle::streams[sid]>>> (dst, numElements, unary);
}

template<typename Dtype, class UnaryEngine>
__global__ void UnaryKernel(Dtype *dst, Dtype* src, int numElements, UnaryEngine unary)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < numElements)
    {
        unary(dst[i], src[i]);
    }
}

template<typename Dtype, class UnaryEngine>
void UnaryOp(Dtype *dst, Dtype* src, int numElements, UnaryEngine unary, const unsigned& sid)
{
    int thread_num = min(c_uCudaThreadNum, numElements);    
    int blocksPerGrid = (numElements + thread_num - 1) / thread_num;
    UnaryKernel<<<blocksPerGrid, thread_num, 0, GPUHandle::streams[sid]>>> (dst, src, numElements, unary);
}

#endif