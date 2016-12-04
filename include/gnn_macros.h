#ifndef GNN_MACROS_H
#define GNN_MACROS_H

#include <cstddef>

namespace gnn
{

#define NOT_IMPLEMENTED { throw std::logic_error(std::string("not implemented virtual func: ") + std::string(__FUNCTION__)); }

typedef unsigned int uint;

struct CPU
{
};

struct GPU
{
};

struct DENSE
{
};

struct SPARSE
{
};

}

#endif