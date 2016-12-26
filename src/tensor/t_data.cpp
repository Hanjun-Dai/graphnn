#include "tensor/t_data.h"
#include "util/mem_holder.h"

namespace gnn
{

template<typename Dtype>
TDataTemplate<CPU, DENSE, Dtype>::TDataTemplate()
		: TData(), ptr(nullptr), mem_size(0)
{

}

template<typename Dtype>
TDataTemplate<CPU, DENSE, Dtype>::~TDataTemplate()
{
	MemHolder<CPU>::DelArr(this->ptr);
}

template<typename Dtype>
void TDataTemplate<CPU, DENSE, Dtype>::Resize(size_t new_size)
{
	if (new_size > this->mem_size)
	{
		this->mem_size = std::max(new_size, this->mem_size * 2);
		MemHolder<CPU>::DelArr(this->ptr);
		MemHolder<CPU>::MallocArr(this->ptr, sizeof(Dtype) * this->mem_size);
	}
}

template class TDataTemplate<CPU, DENSE, float>;
template class TDataTemplate<CPU, DENSE, double>;
template class TDataTemplate<CPU, DENSE, int>;

template<typename mode, typename Dtype>
TDataTemplate<mode, SPARSE, Dtype>::TDataTemplate()
		: TData()
{
	nnz = len_ptr = nzCap = ptrCap = 0;
	val = nullptr;
	col_idx = row_ptr = nullptr;
}

template<typename mode, typename Dtype>
TDataTemplate<mode, SPARSE, Dtype>::~TDataTemplate()
{
	MemHolder<mode>::DelArr(val);
	MemHolder<mode>::DelArr(col_idx);
	MemHolder<mode>::DelArr(row_ptr);	
}

template<typename mode, typename Dtype>
TDataTemplate<mode, SPARSE, Dtype>::TDataTemplate(int newNzCap, int newPtrCap)
		: TData()
{
	nnz = len_ptr = 0;
	nzCap = newNzCap; 
	ptrCap = newPtrCap;
	MemHolder<mode>::MallocArr(val, sizeof(Dtype) * nzCap);
	MemHolder<mode>::MallocArr(col_idx, sizeof(int) * nzCap);
	MemHolder<mode>::MallocArr(row_ptr, sizeof(int) * ptrCap);
}

template class TDataTemplate<CPU, SPARSE, float>;
template class TDataTemplate<CPU, SPARSE, double>;
template class TDataTemplate<CPU, SPARSE, int>;
template class TDataTemplate<GPU, SPARSE, float>;
template class TDataTemplate<GPU, SPARSE, double>;
template class TDataTemplate<GPU, SPARSE, int>;
}