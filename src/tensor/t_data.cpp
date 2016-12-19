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
void TDataTemplate<CPU, DENSE, Dtype>::Resize(size_t new_size)
{
	if (new_size > this->mem_size)
	{
		this->mem_size = new_size;
		MemHolder<CPU>::DelArr(this->ptr);
		MemHolder<CPU>::MallocArr(this->ptr, sizeof(Dtype) * new_size);
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
	col_idx = ptr = nullptr;
}

template class TDataTemplate<CPU, SPARSE, float>;
template class TDataTemplate<CPU, SPARSE, double>;
template class TDataTemplate<CPU, SPARSE, int>;

}