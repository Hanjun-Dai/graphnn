#ifndef GNN_MACROS_H
#define GNN_MACROS_H

#include <cstddef>

namespace gnn
{

#define NOT_IMPLEMENTED { throw std::logic_error(std::string("not implemented virtual func: ") + std::string(__FUNCTION__)); }

typedef unsigned int uint;

enum Mode
{
	CPU = 0,
	GPU = 1
};

enum MatType
{
	DENSE,
	SPARSE
};

enum DataType
{
	FLOAT32,
	FLOAT64,
	INT32
};


}

#endif