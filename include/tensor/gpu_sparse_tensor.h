#ifndef GPU_SPARSE_TENSOR_H
#define GPU_SPARSE_TENSOR_H

#include "tensor.h"
#include "t_data.h"

namespace gnn
{

/**
 * @brief      GPU SPARSE specialization of Tensor
 *
 * @tparam     Dtype  { float/double }
 */
template<typename Dtype>
class TensorTemplate<GPU, SPARSE, Dtype> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;
	
	/**
	 * the shared ptr to the data structure (which is used to keep the data of this tensor)
	 */
	std::shared_ptr< SparseData<GPU, Dtype> > data;
};

/**
 * @brief      GPU SPARSE int tensor specialization; this tensor is not used for heavy computation
 * 				(e.g., matmul)
 */
template<>
class TensorTemplate<GPU, SPARSE, int> : public Tensor
{
public:

	TensorTemplate();

	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;

	/**
	 * the shared ptr to the data structure (which is used to keep the data of this tensor)
	 */	
	std::shared_ptr< SparseData<GPU, int> > data;
};

}


#endif