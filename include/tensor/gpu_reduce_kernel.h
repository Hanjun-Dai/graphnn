#ifndef GPU_REDUCE_KERNEL_H
#define GPU_REDUCE_KERNEL_H

#ifdef USE_GPU

#define REDUCE_THREADS 256
#define REDUCE_THREAD_BITS 8

namespace gnn
{

template<typename Dtype>
class MaxIdxReduce
{
public:
	MaxIdxReduce() {}

	__device__ inline void Reduce(int& dst, const int& x, const Dtype* data)
	{
		if (data[x] > data[dst])
			dst = x;
	}

	__device__ inline int Map(const int& x, const Dtype* data)
	{
		return x;
	}

	__device__ inline void Init(int& dst, const int& x, const Dtype* data)
	{
		dst = x;
	}
};

template<typename Dtype>
class SumReduce
{
public:
	SumReduce() {}

	__device__ inline void Reduce(Dtype& dst, const Dtype& x, const Dtype* data)
	{
		dst += x;
	}

	__device__ inline Dtype Map(const int& x, const Dtype* data)
	{
		return data[x];
	}

	__device__ inline void Init(Dtype& dst, const int& x, const Dtype* data)
	{
		dst = data[x];
	}
};

template<typename dstDtype, typename srcDtype, typename Functor>
__global__ void MatColReduceKernel(dstDtype* dst, srcDtype *orig_ptr, Functor f, int cols)
{
    __shared__ dstDtype buffer[REDUCE_THREADS];

    srcDtype* row_ptr = orig_ptr + blockIdx.x * cols;

    int i_start = threadIdx.x;
    int i_end = cols;
    int i_step = blockDim.x;
	if (i_start < cols)
		f.Init(buffer[threadIdx.x], i_start, row_ptr);
	for (int i = i_start + i_step; i < i_end; i += i_step)
    {
    	f.Reduce(buffer[threadIdx.x], f.Map(i, row_ptr), row_ptr);
    }
    __syncthreads();

    int shift;
    for (int i = REDUCE_THREAD_BITS - 1; i >= 0; --i)
    {
    	shift = 1 << i;
    	if (threadIdx.x < shift && threadIdx.x + shift < cols)
    	{
    		f.Reduce(buffer[threadIdx.x], buffer[threadIdx.x + shift], row_ptr);
    	}
		__syncthreads();
    }
    if (threadIdx.x == 0)
    	dst[blockIdx.x] = buffer[0];
}

class MatColReduce
{
public:
	template<typename dstDtype, typename srcDtype, typename Functor>
	static void Exec(dstDtype* dst, const srcDtype* src, size_t rows, size_t cols, Functor f)
	{
		dim3 blocks(rows);
		dim3 threads(REDUCE_THREADS);
    	MatColReduceKernel<<<blocks, threads, 0, cudaStreamPerThread>>> (dst, src, f, cols);
	}
};
}

#endif

#endif