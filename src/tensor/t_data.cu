#include "tensor/t_data.h"
#include "util/mem_holder.h"

namespace gnn
{

template<typename Dtype>
TDataTemplate<GPU, DENSE, Dtype>::TDataTemplate()
		: ptr(nullptr), mem_size(0)
{

}

template<typename Dtype>
TDataTemplate<GPU, DENSE, Dtype>::~TDataTemplate()
{
	MemHolder<GPU>::DelArr(this->ptr);
}

template<typename Dtype>
void TDataTemplate<GPU, DENSE, Dtype>::Resize(size_t new_size)
{
	if (new_size > this->mem_size)
	{
		this->mem_size = new_size > this->mem_size * 2 ? new_size : this->mem_size * 2;
		MemHolder<GPU>::DelArr(this->ptr);
		MemHolder<GPU>::MallocArr(this->ptr, sizeof(Dtype) * this->mem_size);
		cudaMemset(this->ptr, 0, sizeof(Dtype) * this->mem_size);
		dev_ptr = thrust::device_pointer_cast(ptr);
	}
}

template class TDataTemplate<GPU, DENSE, float>;
template class TDataTemplate<GPU, DENSE, double>;
template class TDataTemplate<GPU, DENSE, int>;

}
