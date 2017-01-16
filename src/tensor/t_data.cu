#include "tensor/t_data.h"
#include "util/mem_holder.h"

namespace gnn
{

template<typename Dtype>
TDataTemplate<GPU, DENSE, Dtype>::TDataTemplate()
		: TData(), ptr(nullptr), mem_size(0), is_referring(false)
{

}

template<typename Dtype>
TDataTemplate<GPU, DENSE, Dtype>::TDataTemplate(Dtype* src_ptr, size_t offset, size_t _msize)
		: TData(), is_referring(true)
{
	this->mem_size = _msize;
	this->ptr = src_ptr + offset;
}

template<typename Dtype>
TDataTemplate<GPU, DENSE, Dtype>::~TDataTemplate()
{
	if (!is_referring)
		MemHolder<GPU>::Recycle(this->ptr);
}

template<typename Dtype>
void TDataTemplate<GPU, DENSE, Dtype>::Resize(size_t new_size)
{	
	if (new_size > this->mem_size)
	{
		ASSERT(!is_referring, "cannot modify view only tensor");
		this->mem_size = new_size;
		MemHolder<GPU>::ForceDel(this->ptr);
		MemHolder<GPU>::MallocArr(this->ptr, sizeof(Dtype) * this->mem_size);
		cudaMemset(this->ptr, 0, sizeof(Dtype) * this->mem_size);
		dev_ptr = thrust::device_pointer_cast(ptr);
	}
}

template class TDataTemplate<GPU, DENSE, float>;
template class TDataTemplate<GPU, DENSE, double>;
template class TDataTemplate<GPU, DENSE, int>;

}
