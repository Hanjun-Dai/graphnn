#ifndef T_DATA_H
#define T_DATA_H

#include "tensor.h"

namespace gnn{

template<Mode mode, MatType mType, DataType dType>
class TDataTemplate;

template<Mode mode, MatType mType, DataType dType>
class TensorTemplate;

template<Mode mode, DataType dType>
using DenseData = TDataTemplate<mode, DENSE, dType>;

template<Mode mode, DataType dType>
using SparseData = TDataTemplate<mode, SPARSE, dType>;

class TData
{
public: 
	template<Mode mode, MatType mType, DataType dType>
	TDataTemplate<mode, mType, dType>& Derived(TensorTemplate<mode, mType, dType>& tensor);

};

template<Mode mode, DataType dType>
class TDataTemplate<mode, DENSE, dType> : public TData
{
public:
	typedef typename std::conditional<dType == FLOAT32, float, typename std::conditional<dType == FLOAT64, double, int>::type >::type D;

	TDataTemplate() : data(nullptr), mem_size(0) {}

	D* data;
	size_t mem_size;
};

template<Mode mode, DataType dType>
class TDataTemplate<mode, SPARSE, dType> : public TData
{
public:
	typedef typename std::conditional<dType == FLOAT32, float, typename std::conditional<dType == FLOAT64, double, int>::type >::type D;

	D* val;
	int* col_idx;
	int* ptr;
	
	int nnz;
	int len_ptr;
};

}

#endif