#ifndef T_DATA_H
#define T_DATA_H

#include "tensor.h"

namespace gnn{

template<typename mode, typename mType, typename Dtype>
class TDataTemplate;

template<typename mode, typename mType, typename Dtype>
class TensorTemplate;

template<typename mode, typename Dtype>
using DenseData = TDataTemplate<mode, DENSE, Dtype>;

template<typename mode, typename Dtype>
using SparseData = TDataTemplate<mode, SPARSE, Dtype>;

class TData
{
public: 
	template<typename mode, typename mType, typename Dtype>
	TDataTemplate<mode, mType, Dtype>& Derived(TensorTemplate<mode, mType, Dtype>* host)
	{
		return *(dynamic_cast<TDataTemplate<mode, mType, Dtype>*>(this));
	}

	template<typename mode, typename mType, typename Dtype>
	TDataTemplate<mode, mType, Dtype>& Derived(TensorTemplate<mode, mType, Dtype>& host)
	{
		return *(dynamic_cast<TDataTemplate<mode, mType, Dtype>*>(this));
	}

	template<typename mode, typename mType, typename Dtype>
	TDataTemplate<mode, mType, Dtype>& Derived()
	{
		return *(dynamic_cast<TDataTemplate<mode, mType, Dtype>*>(this));
	}

private:
	virtual void dummy() {};
};

template<typename mode, typename Dtype>
class TDataTemplate<mode, DENSE, Dtype> : public TData
{
public:

	TDataTemplate() : ptr(nullptr), mem_size(0) {}

	Dtype* ptr;
	size_t mem_size;
};

template<typename mode, typename Dtype>
class TDataTemplate<mode, SPARSE, Dtype> : public TData
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