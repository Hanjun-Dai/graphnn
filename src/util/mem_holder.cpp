#include "util/mem_holder.h"
#include <memory>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>

#include <iostream>

namespace gnn
{

template<typename mode, typename Dtype>
void Malloc(Dtype*& p, size_t nBytes)
{
	if (mode::type == MatMode::cpu)
		p = (Dtype*) malloc(nBytes);
	else {
		cudaError_t t = cudaMalloc(&p, nBytes);
		assert(t != cudaErrorMemoryAllocation);			
	}
}

template< typename mode >
std::multimap<size_t, void*> MemHolder<mode>::avail_pt_map;

template< typename mode >
std::map< std::uintptr_t, std::pair< size_t, void* > >  MemHolder<mode>::pt_info;

template< typename mode >
std::mutex MemHolder<mode>::r_loc;

template<typename mode>
template<typename T>
void MemHolder<mode>::DelArr(T*& p)
{
	r_loc.lock();
	if (p)
	{
		auto id = reinterpret_cast<std::uintptr_t>(p);
		if (pt_info.count(id))
			avail_pt_map.insert(std::make_pair(pt_info[id].first, (void*)p));
		p = nullptr;
	}
	r_loc.unlock();
}

template<typename mode>
template<typename T>
void MemHolder<mode>::MallocArr(T*& p, size_t nBytes)
{
	r_loc.lock();
	if (nBytes)
	{
		auto it = avail_pt_map.lower_bound(nBytes);
		if (it == avail_pt_map.end()) // no available pointer found
		{
			Malloc<mode>(p, nBytes);
			auto id = reinterpret_cast<std::uintptr_t>(p);
			ASSERT(pt_info.count(id) == 0, "pointer duplicates");
			if (mode::type == MatMode::gpu)
				std::cerr << "allocate gpu" << id << std::endl;
			else{
				std::cerr << "allocate cpu" << id << std::endl;
			// 	void *array[10];
   //          size_t size;
   //          size = backtrace(array, 10);
   //          backtrace_symbols_fd(array, size, STDERR_FILENO); 
			}
			pt_info[id] = std::make_pair(nBytes, (void*)p);
		} else {
			p = (T*)it->second;
			avail_pt_map.erase(it);
		}
	}
	else p = nullptr;
	r_loc.unlock();
}

template<typename mode>
void MemHolder<mode>::Clear()
{
	for (auto p : pt_info)
	{
		if (mode::type == MatMode::cpu)
			free(p.second.second);
		else
			cudaFree(p.second.second);
	}
	avail_pt_map.clear();
	pt_info.clear();
}

template class MemHolder<CPU>;
template class MemHolder<GPU>;

template void MemHolder<CPU>::DelArr<float>(float*& p); 
template void MemHolder<CPU>::DelArr<double>(double*& p); 
template void MemHolder<CPU>::DelArr<int>(int*& p); 

template void MemHolder<CPU>::MallocArr<float>(float*& p, size_t nBytes); 
template void MemHolder<CPU>::MallocArr<double>(double*& p, size_t nBytes); 
template void MemHolder<CPU>::MallocArr<int>(int*& p, size_t nBytes); 

template void MemHolder<GPU>::DelArr<float>(float*& p); 
template void MemHolder<GPU>::DelArr<double>(double*& p); 
template void MemHolder<GPU>::DelArr<int>(int*& p); 

template void MemHolder<GPU>::MallocArr<float>(float*& p, size_t nBytes); 
template void MemHolder<GPU>::MallocArr<double>(double*& p, size_t nBytes); 
template void MemHolder<GPU>::MallocArr<int>(int*& p, size_t nBytes); 

}
