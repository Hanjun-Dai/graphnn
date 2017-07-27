#include "tensor/t_data.h"
#include "util/mem_holder.h"

namespace gnn
{

template<typename Dtype>
TDataTemplate<CPU, DENSE, Dtype>::TDataTemplate()
		: TData(), ptr(nullptr), mem_size(0), is_referring(false)
{

}

template<typename Dtype>
TDataTemplate<CPU, DENSE, Dtype>::TDataTemplate(Dtype* src_ptr, size_t offset, size_t _msize)
		: TData(), is_referring(true)
{
	this->mem_size = _msize;
	this->ptr = src_ptr + offset;
}

template<typename Dtype>
TDataTemplate<CPU, DENSE, Dtype>::~TDataTemplate()
{
	if (!is_referring)
		MemHolder<CPU>::Recycle(this->ptr);
}

template<typename Dtype>
void TDataTemplate<CPU, DENSE, Dtype>::Resize(size_t new_size)
{	
	if (new_size > this->mem_size)
	{
		ASSERT(!is_referring, "cannot modify view only tensor");
		this->mem_size = new_size;
		MemHolder<CPU>::ForceDel(this->ptr);
		MemHolder<CPU>::MallocArr(this->ptr, sizeof(Dtype) * this->mem_size);
	}
}

template class TDataTemplate<CPU, DENSE, float>;
template class TDataTemplate<CPU, DENSE, double>;
template class TDataTemplate<CPU, DENSE, int>;

template<typename mode, typename Dtype>
TDataTemplate<mode, CSR_SPARSE, Dtype>::TDataTemplate()
		: TData()
{
	nnz = len_ptr = nzCap = ptrCap = 0;
	val = nullptr;
	col_idx = row_ptr = nullptr;
}

template<typename mode, typename Dtype>
TDataTemplate<mode, CSR_SPARSE, Dtype>::~TDataTemplate()
{
	MemHolder<mode>::Recycle(val);
	MemHolder<mode>::Recycle(col_idx);
	MemHolder<mode>::Recycle(row_ptr);	
}

template<typename mode, typename Dtype>
TDataTemplate<mode, CSR_SPARSE, Dtype>::TDataTemplate(int newNzCap, int newPtrCap)
		: TData()
{
	nnz = len_ptr = 0;
	nzCap = newNzCap; 
	ptrCap = newPtrCap;
	MemHolder<mode>::MallocArr(val, sizeof(Dtype) * nzCap);
	MemHolder<mode>::MallocArr(col_idx, sizeof(int) * nzCap);
	MemHolder<mode>::MallocArr(row_ptr, sizeof(int) * ptrCap);
}

template class TDataTemplate<CPU, CSR_SPARSE, float>;
template class TDataTemplate<CPU, CSR_SPARSE, double>;
template class TDataTemplate<CPU, CSR_SPARSE, int>;
#ifdef USE_GPU
template class TDataTemplate<GPU, CSR_SPARSE, float>;
template class TDataTemplate<GPU, CSR_SPARSE, double>;
template class TDataTemplate<GPU, CSR_SPARSE, int>;
#endif

}