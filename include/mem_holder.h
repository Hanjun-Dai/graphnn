#ifndef MEM_HOLDER_H
#define MEM_HOLDER_H

#include "gnn_macros.h"

namespace gnn
{

template< Mode mode >
class MemHolder
{
public:
	
	template<typename T>
	static void DelArr(T*& p);

	template<typename T>
	static void MallocArr(T*& p, size_t nBytes);
};

}

#endif