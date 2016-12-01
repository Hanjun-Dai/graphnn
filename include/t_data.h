#ifndef T_DATA_H
#define T_DATA_H

#include "tensor.h"

namespace gnn{

class TData
{
public: 
	
};


template<Mode mode, typename Dtype>
class SparseData : public TData
{
public:
	Dtype* data;
	size_t mem_size;
};

template<Mode mode, typename Dtype>
class DenseData : public TData
{
public:

	Dtype* val;
	int* col_idx;
	int* ptr;
	
	int nnz;
	int len_ptr;
};

}

#endif