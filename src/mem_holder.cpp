#include "mem_holder.h"
#include <memory>
#include <cstdlib>

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

}
