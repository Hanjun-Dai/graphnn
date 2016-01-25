#ifndef MAT_TYPEDEF_H
#define MAT_TYPEDEF_H

#include <cstddef>

enum MatMode
{
	CPU = 0,
	GPU = 1
};

enum class Trans
{
	N = 0,
	T = 1
};

enum Phase
{
    TRAIN = 0,
    TEST = 1  
};

enum MatType
{
	DENSE,
	SPARSE
};

#define c_uCudaThreadNum 1024

const double eps = 1e-8;

#endif