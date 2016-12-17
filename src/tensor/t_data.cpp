#include "tensor/t_data.h"
#include "util/mem_holder.h"

namespace gnn
{

template<typename Dtype>
TDataTemplate<CPU, DENSE, Dtype>::TDataTemplate()
		: ptr(nullptr), mem_size(0)
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

}