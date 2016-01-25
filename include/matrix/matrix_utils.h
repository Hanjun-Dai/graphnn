#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include "mat_typedef.h"
#include <cuda_runtime.h>
#include <mkl.h>
#include <cublas_v2.h>
#include <memory>
#include <cassert>
#include <iostream>

template<MatMode mode>
struct MatUtils
{
	template<typename T>
	static void DelArr(T*& p);

	template<typename T>
	static void MallocArr(T*& p, size_t nBytes);

	template<typename T>
	static void ArrSetZeros(T*& p, size_t nBytes);
};

template<>
template<typename T>
void MatUtils<CPU>::ArrSetZeros(T*& p, size_t nBytes)
{
	if (p)
		memset(p, 0, nBytes);
}

template<>
template<typename T>
void MatUtils<CPU>::DelArr(T*& p)
{
	if (p)
	{
		delete[] p; 
		p = nullptr;
	}
}

template<>
template<typename T>
void MatUtils<CPU>::MallocArr(T*& p, size_t nBytes)
{
	if (nBytes)
		p = (T*) malloc(nBytes);
	else p = nullptr;
}

template<>
template<typename T>
void MatUtils<GPU>::DelArr(T*& p)
{
	if (p)
	{		
		cudaFree(p);
		p = nullptr;
	}
}

template<>
template<typename T>
void MatUtils<GPU>::MallocArr(T*& p, size_t nBytes)
{
	if (nBytes)
	{
		cudaError_t t = cudaMalloc(&p, nBytes);
		assert(t != cudaErrorMemoryAllocation);
	}
	else p = nullptr;
}


inline void GetDims(const size_t& lhs_rows, const size_t& lhs_cols, Trans ltrans, 
					const size_t& rhs_rows, const size_t& rhs_cols, Trans rtrans, 
					size_t &m, size_t &n, size_t &k)
{
	m = ltrans == Trans::N ? lhs_rows : lhs_cols;
	n = rtrans == Trans::N ? rhs_cols : rhs_rows;
	k = ltrans == Trans::N ? lhs_cols : lhs_rows;
	assert((rtrans == Trans::N && rhs_rows == k) || (rtrans == Trans::T && rhs_cols == k));
}

#endif