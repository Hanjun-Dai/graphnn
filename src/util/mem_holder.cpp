#include "util/mem_holder.h"
#include <memory>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>

namespace gnn
{

template<>
template<typename T>
void MemHolder<CPU>::DelArr(T*& p)
{
	if (p)
	{
		delete[] p; 
		p = nullptr;
	}
}

template<>
template<typename T>
void MemHolder<CPU>::MallocArr(T*& p, size_t nBytes)
{
	if (nBytes)
		p = (T*) malloc(nBytes);
	else p = nullptr;
}

template void MemHolder<CPU>::DelArr<float>(float*& p); 
template void MemHolder<CPU>::DelArr<double>(double*& p); 
template void MemHolder<CPU>::DelArr<int>(int*& p); 

template void MemHolder<CPU>::MallocArr<float>(float*& p, size_t nBytes); 
template void MemHolder<CPU>::MallocArr<double>(double*& p, size_t nBytes); 
template void MemHolder<CPU>::MallocArr<int>(int*& p, size_t nBytes); 

template<>
template<typename T>
void MemHolder<GPU>::DelArr(T*& p)
{
	if (p)
	{		
		cudaFree(p);
		p = nullptr;
	}
}

template<>
template<typename T>
void MemHolder<GPU>::MallocArr(T*& p, size_t nBytes)
{
	if (nBytes)
	{
		cudaError_t t = cudaMalloc(&p, nBytes);
		assert(t != cudaErrorMemoryAllocation);
	}
	else p = nullptr;
}

template void MemHolder<GPU>::DelArr<float>(float*& p); 
template void MemHolder<GPU>::DelArr<double>(double*& p); 
template void MemHolder<GPU>::DelArr<int>(int*& p); 

template void MemHolder<GPU>::MallocArr<float>(float*& p, size_t nBytes); 
template void MemHolder<GPU>::MallocArr<double>(double*& p, size_t nBytes); 
template void MemHolder<GPU>::MallocArr<int>(int*& p, size_t nBytes); 

}
