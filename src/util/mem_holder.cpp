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
void MemHolder<mode>::Recycle(T*& p)
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
void MemHolder<mode>::ForceDel(T*& p)
{
	r_loc.lock();
	if (p)
	{
		auto id = reinterpret_cast<std::uintptr_t>(p);
		auto it = pt_info.find(id);
		ASSERT(it != pt_info.end(), "pointer not found");
		if (mode::type == MatMode::cpu)
			free(it->second.second);
		else
			cudaFree(it->second.second);
		pt_info.erase(it);
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
			pt_info[id] = std::make_pair(nBytes, (void*)p);
		} else {
			p = (T*)it->second;
			avail_pt_map.erase(it);
			auto id = reinterpret_cast<std::uintptr_t>(p);
			ASSERT(pt_info.count(id), "unknown pointer");
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

template void MemHolder<CPU>::ForceDel<float>(float*& p); 
template void MemHolder<CPU>::ForceDel<double>(double*& p); 
template void MemHolder<CPU>::ForceDel<int>(int*& p); 

template void MemHolder<CPU>::Recycle<float>(float*& p); 
template void MemHolder<CPU>::Recycle<double>(double*& p); 
template void MemHolder<CPU>::Recycle<int>(int*& p); 

template void MemHolder<CPU>::MallocArr<float>(float*& p, size_t nBytes); 
template void MemHolder<CPU>::MallocArr<double>(double*& p, size_t nBytes); 
template void MemHolder<CPU>::MallocArr<int>(int*& p, size_t nBytes); 

template void MemHolder<GPU>::ForceDel<float>(float*& p); 
template void MemHolder<GPU>::ForceDel<double>(double*& p); 
template void MemHolder<GPU>::ForceDel<int>(int*& p); 

template void MemHolder<GPU>::Recycle<float>(float*& p); 
template void MemHolder<GPU>::Recycle<double>(double*& p); 
template void MemHolder<GPU>::Recycle<int>(int*& p); 

template void MemHolder<GPU>::MallocArr<float>(float*& p, size_t nBytes); 
template void MemHolder<GPU>::MallocArr<double>(double*& p, size_t nBytes); 
template void MemHolder<GPU>::MallocArr<int>(int*& p, size_t nBytes); 

}
