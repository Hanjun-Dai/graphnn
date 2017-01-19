#ifndef MEM_HOLDER_H
#define MEM_HOLDER_H

#include "gnn_macros.h"
#include <map>
#include <cstdint>
#include <mutex>

namespace gnn
{

/**
 * @brief      responsible for memory allocation and deletion; 
 * 				the dynamic computation graph will get better performance 
 * 				with persistent memory
 *
 * @tparam     mode  { CPU/GPU }
 */
template< typename mode >
class MemHolder
{
public:
	/**
	 * @brief      recycle an array allocated before (must be the head pointer)
	 *
	 * @param      p     { the head pointer }
	 *
	 * @tparam     T     { the data type }
	 */
	template<typename T>
	static void Recycle(T*& p);

	template<typename T>
	static void ForceDel(T*& p);

	/**
	 * @brief      allocate new array of memory 
	 *
	 * @param      p       { the pointer }
	 * @param[in]  nBytes  # bytes to be allocated
	 *
	 * @tparam     T       { the array data type }
	 */
	template<typename T>
	static void MallocArr(T*& p, size_t nBytes);

	static std::multimap<size_t, void*> avail_pt_map;
	static std::map< std::uintptr_t, std::pair< size_t, void* > > pt_info;
	static std::mutex r_loc;

};

}

#endif