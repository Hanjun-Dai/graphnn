#ifndef T_DATA_H
#define T_DATA_H

#include "tensor.h"

namespace gnn{

template<typename mode, typename mType>
class TDataTemplate;

template<typename mode, typename mType>
class TensorTemplate;

template<typename mode>
using DenseData = TDataTemplate<mode, DENSE>;

template<typename mode>
using SparseData = TDataTemplate<mode, SPARSE>;

class TData
{
public: 
	template<typename mode, typename mType>
	TDataTemplate<mode, mType>& Derived(TensorTemplate<mode, mType>*)
	{
		return *(dynamic_cast<TDataTemplate<mode, mType>*>(this));
	}

	template<typename mode, typename mType>
	TDataTemplate<mode, mType>& Derived(TensorTemplate<mode, mType>&)
	{
		return *(dynamic_cast<TDataTemplate<mode, mType>*>(this));
	}

	template<typename mode, typename mType>
	TDataTemplate<mode, mType>& Derived()
	{
		return *(dynamic_cast<TDataTemplate<mode, mType>*>(this));
	}

private:
	virtual void dummy() {};
};

template<typename mode>
class TDataTemplate<mode, DENSE> : public TData
{
public:

	TDataTemplate() : ptr(nullptr), mem_size(0) {}

	Dtype* ptr;
	size_t mem_size;
};

template<typename mode>
class TDataTemplate<mode, SPARSE> : public TData
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