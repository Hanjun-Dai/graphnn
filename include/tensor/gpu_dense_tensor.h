#ifndef GPU_DENSE_TENSOR_H
#define GPU_DENSE_TENSOR_H

#include "tensor.h"
#include "t_data.h"
#include "gpu_handle.h"

namespace gnn
{

/**
 * @brief      GPU DENSE specialization of tensor
 *
 * @tparam     Dtype   float/double
 */
template<typename Dtype>
class TensorTemplate<GPU, DENSE, Dtype> : public Tensor
{
public:

	TensorTemplate();
	TensorTemplate(std::vector<size_t> l);
	TensorTemplate(TShape s);

	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;

	/**
	 * @brief      deeply copy src to this tensor
	 *
	 * @param      src   the CPU dense tensor with same data type
	 */
	void CopyFrom(DTensor<CPU, Dtype>& src);		
	/**
	 * @brief      deeply copy src to this tensor
	 *
	 * @param      src   the GPU dense tensor with same data type
	 */
	void CopyFrom(DTensor<GPU, Dtype>& src);

	/**
	 * @brief      shallow copy (only set the shared_ptr)
	 *
	 * @param      src   The source
	 */
	void ShallowCopy(DTensor<GPU, Dtype>& src);

	/**
	 * @brief      set this tensor to be zero
	 */
	void Zeros(); 

	/**
	 * @brief      if this tensor is actually a scalar, return it; otherwise raise error
	 *
	 * @return     the only element in this tensor
	 */
	Dtype AsScalar(); 

	/**
	 * @brief      Sets this tensor from random normal
	 *
	 * @param[in]  mean  The mean
	 * @param[in]  std   The standard deviation
	 */
	void SetRandN(Dtype mean, Dtype std);

	/**
	 * @brief      Sets this tensor from random uniform
	 *
	 * @param[in]  lb    The lower bound
	 * @param[in]  ub    The upper bound
	 */
	void SetRandU(Dtype lb, Dtype ub);

	/**
	 * @brief      fill this tensor with same scalar
	 *
	 * @param[in]  scalar  The scalar to be set
	 */
	void Fill(Dtype scalar);

	/**
	 * @brief      the absolute sum of this tensor 
	 *
	 * @return     the absolute sum
	 */
	Dtype ASum();
	
	/**
	 * the shared ptr to the data structure (which is used to keep the data of this tensor)
	 */
	std::shared_ptr< DenseData<GPU, Dtype> > data;
};

/**
 * @brief      GPU DENSE int tensor specialization; this tensor is not used for heavy computation
 * 				(e.g., matmul)
 */
template<>
class TensorTemplate<GPU, DENSE, int> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;
	
	/**
	 * the shared ptr to the data structure (which is used to keep the data of this tensor)
	 */	
	std::shared_ptr< DenseData<GPU, int> > data;
};

}


#endif