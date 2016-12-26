#ifndef GPU_DENSE_TENSOR_H
#define GPU_DENSE_TENSOR_H

#include "tensor.h"
#include "t_data.h"
#include "gpu_handle.h"

namespace gnn
{

template<typename dstType, typename srcType>
void TypeCastCopy(dstType* dst, srcType* src, size_t count);

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
	virtual ~TensorTemplate() {}	
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
	 * @brief      deeply copy src to this tensor
	 *
	 * @param      src        the GPU dense tensor with different data type
	 *
	 * @tparam     otherType  the src's data type
	 */
	template<typename otherType> 
	void CopyFrom(DTensor<GPU, otherType>& src)
	{
		Reshape(src.shape.dims);
		TypeCastCopy(this->data->ptr, src.data->ptr, this->shape.Count());
	}

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
	 * @brief      find the max index along dimensions other than axis
	 *
	 * @param      dst   used to store the results
	 * @param[in]  axis  The axis to be kept
	 */
	void ArgMax(DTensor<GPU, int>& dst, uint axis = 0);	

	/**
	 * @brief      compute and store the result (a dense matrix) of matrix multiplication; 
	 *				this = alpha * A x B + beta * this
	 * @param      a       operand A (dense matrix)
	 * @param      b       operand B (dense matrix)
	 * @param[in]  transA  whether to transpose A
	 * @param[in]  transB  whether to transpose B
	 * @param[in]  alpha   The alpha
	 * @param[in]  beta    The beta
	 */
	void MM(DTensor<GPU, Dtype>& a, DTensor<GPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta);

	/**
	 * @brief      compute and store the result (a dense matrix) of matrix multiplication; 
	 *				this = alpha * A x B + beta * this
	 * @param      a       operand A (sparse matrix)
	 * @param      b       operand B (dense matrix)
	 * @param[in]  transA  whether to transpose A
	 * @param[in]  transB  whether to transpose B
	 * @param[in]  alpha   The alpha
	 * @param[in]  beta    The beta
	 */
	void MM(SpTensor<GPU, Dtype>& a, DTensor<GPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta);

	/**
	 * @brief      do the softmax inplace, keep rows
	 */
	void Softmax();

	/**
	 * @brief      store the result of mean reduction
	 *
	 * @param      a    the operand
	 * @param[in]  axis  The axis to be reduced; by default it is -1, which will do global reduce
	 */
	void Mean(DTensor<GPU, Dtype>& a, int axis = -1);		

	/**
	 * @brief      Add a scalar to each element
	 *
	 * @param[in]  scalar  The scalar to be added
	 */
	void Add(Dtype scalar);

	/**
	 * @brief      the same axpy defined in blas: y = a * x + y
	 *
	 * @param[in]  a     scalar a
	 * @param      x     dense tensor x
	 */
	void Axpy(Dtype a, DTensor<GPU, Dtype>& x);

	/**
	 * @brief      the same axpy defined in blas: y = a * x + y; but here
	 * 				x is a sparse tensor
	 *
	 * @param[in]  a     scalar a
	 * @param      x     sparse tensor x
	 */
	void Axpy(Dtype a, SpTensor<GPU, Dtype>& x);

	/**
	 * @brief      the same axpby defined in blas: y = a * x + b * y
	 *
	 * @param[in]  a     scalar a
	 * @param      x     dense tensor x
	 * @param[in]  b     scalar b
	 */
	void Axpby(Dtype a, DTensor<GPU, Dtype>& x, Dtype b);

	/**
	 * @brief      element-wise multiplication between dense and sparse tensor
	 * 				the same shape of two tensors is required.
	 *
	 * @param      src   The sparse tensor
	 */
	void ElewiseMul(SpTensor<GPU, Dtype>& src);

	/**
	 * @brief      element-wise multiplication between two dense tensors; broadcasting
	 * 				is enabled, but we assume the result tensor keeps the shape of this
	 * 				current tensor (caller).
	 *
	 * @param      src   The other dense tensor
	 */
	void ElewiseMul(DTensor<GPU, Dtype>& src);

	/**
	 * @brief      multipy the tensor with a scalar
	 *
	 * @param[in]  scalar  The scalar to be multiplied
	 */
	void Scale(Dtype scalar);

	/**
	 * @brief      set each element x to be 1/x
	 */
	void Inv();

	/**
	 * @brief      get l2-norm of tensor
	 *
	 * @return     norm2 scalar
	 */
	Dtype Norm2();

	/**
	 * @brief      set each element x to be x^2
	 */
	void Square();
	/**
	 * @brief      set each element x to be x^0.5
	 */
	void Sqrt();
			
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
	virtual ~TensorTemplate() {}
	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;
	/**
	 * @brief      deeply copy src to this tensor
	 *
	 * @param      src   the CPU dense tensor with same data type
	 */
	void CopyFrom(DTensor<CPU, int>& src);		
	/**
	 * @brief      deeply copy src to this tensor
	 *
	 * @param      src   the GPU dense tensor with same data type
	 */
	void CopyFrom(DTensor<GPU, int>& src);

	/**
	 * @brief      shallow copy (only set the shared_ptr)
	 *
	 * @param      src   The source
	 */
	void ShallowCopy(DTensor<GPU, int>& src);	
	
	/**
	 * @brief      set this tensor to be zero
	 */
	void Zeros(); 

	/**
	 * @brief      if this tensor is actually a scalar, return it; otherwise raise error
	 *
	 * @return     the only element in this tensor
	 */
	int AsScalar(); 

	/**
	 * @brief      fill this tensor with same scalar
	 *
	 * @param[in]  scalar  The scalar to be set
	 */
	void Fill(int scalar);

	/**
	 * the shared ptr to the data structure (which is used to keep the data of this tensor)
	 */	
	std::shared_ptr< DenseData<GPU, int> > data;
};

}


#endif