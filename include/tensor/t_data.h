#ifndef T_DATA_H
#define T_DATA_H

#include "tensor.h"
#ifdef USE_GPU
#include <thrust/device_vector.h>
#endif

namespace gnn{

/**
 * @brief      Class for data template.
 *
 * @tparam     mode   { description }
 * @tparam     mType  { description }
 * @tparam     Dtype  { description }
 */
template<typename mode, typename mType, typename Dtype>
class TDataTemplate;

template<typename mode, typename Dtype>
using DenseData = TDataTemplate<mode, DENSE, Dtype>;

template<typename mode, typename Dtype>
using SparseData = TDataTemplate<mode, CSR_SPARSE, Dtype>;

/**
 * @brief      the data object that used to keep the values of tensor
 */
class TData
{
public:
	virtual ~TData() {}
};

/**
 * @brief      CPU DENSE specialization
 *
 * @tparam     Dtype  { float/double/int }
 */
template<typename Dtype>
class TDataTemplate<CPU, DENSE, Dtype> : public TData
{
public:

	TDataTemplate();
	TDataTemplate(Dtype* src_ptr, size_t offset, size_t _msize);

	virtual ~TDataTemplate();
	/**
	 * @brief      resize the allocated memory; only when the new size is 
	 * 				larger than the old one, will we do memory realloc
	 *
	 * @param[in]  new_size  the new size
	 */
	void Resize(size_t new_size);

	/**
	 * the pointer to the memory space
	 */
	Dtype* ptr;
	/**
	 * the memory size
	 */
	size_t mem_size;

	/**
	 * whether this data struct is referring to others
	 */
	bool is_referring;
};

#ifdef USE_GPU
/**
 * @brief      GPU DENSE specialization
 *
 * @tparam     Dtype  { float/double/int }
 */
template<typename Dtype>
class TDataTemplate<GPU, DENSE, Dtype> : public TData
{
public:

	TDataTemplate();
	TDataTemplate(Dtype* src_ptr, size_t offset, size_t _msize);

	virtual ~TDataTemplate();
	/**
	 * @brief      resize the allocated memory; only when the new size is 
	 * 				larger than the old one, will we do memory realloc
	 *
	 * @param[in]  new_size  the new size
	 */	
	void Resize(size_t new_size);
	/**
	 * the pointer to the memory space
	 */
	Dtype* ptr;

	/**
	 * thrust wrapper of ptr
	 */
	thrust::device_ptr<Dtype> dev_ptr;
	/**
	 * the memory size
	 */	
	size_t mem_size;
	
	/**
	 * whether this data struct is referring to others
	 */
	bool is_referring;	
};
#endif

/**
 * @brief      CSR_SPARSE specialization of tensor data object
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double/int }
 */
template<typename mode, typename Dtype>
class TDataTemplate<mode, CSR_SPARSE, Dtype> : public TData
{
public:
	TDataTemplate();
	virtual ~TDataTemplate();
	/**
	 * @brief      constructor
	 *
	 * @param[in]  newNzCap   The new nnz capability
	 * @param[in]  newPtrCap  The new row pointer capability
	 */
	TDataTemplate(int newNzCap, int newPtrCap); 

	/**
	 * actual value
	 */
	Dtype* val;
	/**
	 * column index in CSR format
	 */
	int* col_idx;
	/**
	 * row pointer in CSR format
	 */
	int* row_ptr;
	
	/**
	 * # nonzeros
	 */
	int nnz;
	/**
	 * n_rows + 1
	 */
	int len_ptr;
	/**
	 * maximum nnz (length of val and col_idx)
	 */
	int nzCap;
	/**
	 * maximum row pointer length
	 */
	int ptrCap;	
};

}

#endif
