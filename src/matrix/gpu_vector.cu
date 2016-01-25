#include "vector.h"
#include "cuda_unary_kernel.cuh"
#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <iostream>

template<typename Dtype>
Vector<GPU, Dtype>::~Vector()
{
	MatUtils<GPU>::DelArr(data);
}

template<typename Dtype>
Vector<GPU, Dtype>::Vector()
{
	count = mem_size = 0U;
	streamid = 0U;
	data = nullptr;
}

template<typename Dtype>
Vector<GPU, Dtype>::Vector(size_t _count, unsigned int _streamid)
{	
	count = _count;
	mem_size = count + 1;
	MatUtils<GPU>::MallocArr(data, sizeof(Dtype) * mem_size);			
	streamid = _streamid;
}

template<typename Dtype>
void Vector<GPU, Dtype>::Resize(size_t _count)
{	
	count = _count;
	if (count > mem_size)
	{
		mem_size = count + 1;
		MatUtils<GPU>::DelArr(data);
		MatUtils<GPU>::MallocArr(data, sizeof(Dtype) * mem_size);
	}
}

template<typename Dtype>
void Vector<GPU, Dtype>::Fill(Dtype scalar)
{	
	if (fabs(scalar) < eps)
		cudaMemset(data, 0, count * sizeof(Dtype));
	else
        UnaryOp(data, count, UnarySet<Dtype>(scalar), streamid);
}

template<typename Dtype>
void Vector<GPU, Dtype>::CopyFrom(Vector<CPU, Dtype>& src)
{
	Resize(src.count);
	cudaMemcpyAsync(data, src.data, sizeof(Dtype) * count, cudaMemcpyHostToDevice, GPUHandle::streams[streamid]);
}

template class Vector<GPU, double>;
template class Vector<GPU, float>;