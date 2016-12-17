#ifndef GNN_MACROS_H
#define GNN_MACROS_H

#include <cstddef>

namespace gnn
{

#define NOT_IMPLEMENTED { throw std::logic_error(std::string("not implemented virtual func: ") + std::string(__FUNCTION__)); }

typedef unsigned int uint;

enum class MatType
{
	dense,
	sparse
};

enum class MatMode
{
	cpu = 0,
	gpu = 1
};

struct CPU
{
	static const MatMode mode = MatMode::cpu;
};

struct GPU
{
	static const MatMode mode = MatMode::gpu;
};

struct DENSE
{
	static const MatType type = MatType::dense;
};

struct SPARSE
{
	static const MatType type = MatType::sparse;
};

}

#endif