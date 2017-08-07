#ifndef CPU_DENSE_TENSOR_H
#define CPU_DENSE_TENSOR_H

#include "tensor.h"
#include "t_data.h"

namespace gnn
{

/**
 * @brief      CPU DENSE specialization of tensor
 *
 * @tparam     Dtype   float/double
 */
template<typename Dtype>
class TensorTemplate<CPU, DENSE, Dtype> : public Tensor
{
public:

	TensorTemplate();
	virtual ~TensorTemplate() {}
	TensorTemplate(std::vector<size_t> l, Dtype* _data = nullptr);
	TensorTemplate(TShape s, Dtype* _data = nullptr);
	
	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;

	/**
	 * @brief      save to disk
	 *
	 * @param      fid   The file handle
	 */
	virtual void Serialize(FILE* fid) override;

	/**
	 * @brief      load from disk
	 *
	 * @param      fid   The file handle
	 */
	virtual void Deserialize(FILE* fid) override;
	
	/**
	 * @brief      deeply copy src to this tensor
	 *
	 * @param      src   the CPU dense tensor with same data type
	 */
	void CopyFrom(DTensor<CPU, Dtype>& src);		
#ifdef USE_GPU
	/**
	 * @brief      deeply copy src to this tensor
	 *
	 * @param      src   the GPU dense tensor with same data type
	 */
	void CopyFrom(DTensor<GPU, Dtype>& src);
#endif
	/**
	 * @brief      deeply copy src to this tensor
	 *
	 * @param      src        the CPU dense tensor with different data type
	 *
	 * @tparam     otherType  the src's data type
	 */
	template<typename otherType> 
	void CopyFrom(DTensor<CPU, otherType>& src)
	{
		Reshape(src.shape.dims);
		for (size_t i = 0; i < src.shape.Count(); ++i)
			data->ptr[i] = src.data->ptr[i];
	}

	/**
	 * @brief      shallow copy (only set the shared_ptr)
	 *
	 * @param      src   The source
	 */
	void ShallowCopy(DTensor<CPU, Dtype>& src);

	/**
	 * @brief      get reference of rows
	 *
	 * @param[in]  row_start  The row start
	 * @param[in]  row_cnt    The row count
	 */
	DTensor<CPU, Dtype> GetRowRef(size_t row_start, size_t row_cnt);
	
	/**
	 * @brief      set this tensor to be zero
	 */
	void Zeros(); 

	/**
	 * @brief      set selected rows to be zero
	 *
	 * @param      row_idxes  The row idxes
	 */
	void RowSelectiveZeros(DTensor<CPU, int>& row_idxes);

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
	void ArgMax(DTensor<CPU, int>& dst, uint axis = 0);

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
	void MM(DTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta);

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
	void MM(SpTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta);

	/**
	 * @brief      do the softmax inplace, keep rows
	 */
	void Softmax();

	/**
	 * @brief      inplace jagged softmax; apply to column vector
	 *
	 * @param      lens  The lens of each segment
	 */
	void JaggedSoftmax(DTensor<CPU, int>& lens);

	/**
	 * @brief      store the result of sum reduction
	 *
	 * @param      a    the operand
	 * @param[in]  axis  The axis to be reduced; by default it is -1, which will do global reduce
	 */
	void Sum(DTensor<CPU, Dtype>& a, int axis = -1);
	
	/**
	 * @brief      store the result of mean reduction
	 *
	 * @param      a    the operand
	 * @param[in]  axis  The axis to be reduced; by default it is -1, which will do global reduce
	 */
	void Mean(DTensor<CPU, Dtype>& a, int axis = -1);

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
	void Axpy(Dtype a, DTensor<CPU, Dtype>& x);


	/**
	 * @brief      the same axpy defined in blas: y = a * x + y, but on
	 * 				selected rows
	 *
	 * @param      row_idxes  The row idxes
	 * @param[in]  a          scalar a
	 * @param      x          dense tensor x
	 */
	void RowSelectiveAxpy(DTensor<CPU, int>& row_idxes, Dtype a, DTensor<CPU, Dtype>& x);

	/**
	 * @brief      the same axpy defined in blas: y = a * x + y; but here
	 * 				x is a sparse tensor
	 *
	 * @param[in]  a     scalar a
	 * @param      x     sparse tensor x
	 */
	void Axpy(Dtype a, SpTensor<CPU, Dtype>& x);

	/**
	 * @brief      the same axpby defined in blas: y = a * x + b * y
	 *
	 * @param[in]  a     scalar a
	 * @param      x     dense tensor x
	 * @param[in]  b     scalar b
	 */
	void Axpby(Dtype a, DTensor<CPU, Dtype>& x, Dtype b);

	/**
	 * @brief      the same axpby defined in blas: y = a * x + b * y
	 *
	 * @param[in]  a     scalar a
	 * @param      x     row sparse tensor x
	 * @param[in]  b     scalar b
	 */
	void RowSparseAxpby(Dtype a, RowSpTensor<CPU, Dtype>& x, Dtype b);

	/**
	 * @brief      broadcast operation
	 *
	 * @param      src   The rhs operator
	 * @param[in]  opr   The actual calculation
	 */
	void BCast(DTensor<CPU, Dtype>& src, std::function<void(Dtype&, Dtype&)> opr);

	/**
	 * @brief      concatenate cols of {matrix}
	 *
	 * @param[in]  src_list  The matrix list
	 */
	void ConcatCols(std::vector< DTensor<CPU, Dtype>* > src_list);

	/**
	 * @brief      copy cols from src
	 *
	 * @param      src        The source matrix
	 * @param[in]  col_start  The col start
	 * @param[in]  col_cnt    The col count
	 */
	void CopyColsFrom(DTensor<CPU, Dtype>& src, size_t col_start, size_t col_cnt); 

	/**
	 * @brief      element-wise multiplication between dense and sparse tensor
	 * 				the same shape of two tensors is required.
	 *
	 * @param      src   The sparse tensor
	 */
	void ElewiseMul(SpTensor<CPU, Dtype>& src);

	/**
	 * @brief      element-wise multiplication between two dense tensors; broadcasting
	 * 				is enabled, but we assume the result tensor keeps the shape of this
	 * 				current tensor (caller).
	 *
	 * @param      src   The other dense tensor
	 */
	void ElewiseMul(DTensor<CPU, Dtype>& src);

	/**
	 * @brief      element-wise division between two dense tensors; broadcasting
	 * 				is enabled, but we assume the result tensor keeps the shape of this
	 * 				current tensor (caller).
	 *
	 * @param      src   The other dense tensor
	 */
	void ElewiseDiv(DTensor<CPU, Dtype>& src);	

	/**
	 * @brief      multipy the tensor with a scalar
	 *
	 * @param[in]  scalar  The scalar to be multiplied
	 */
	void Scale(Dtype scalar);

	/**
	 * @brief      set each element x to be |x|
	 */
	void Abs();

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
	 * @brief      set each element x to be 1/ (x^0.5)
	 */
	void InvSqrt();

	/**
	 * @brief      set each element x to be 1 / (1 + exp(-x))
	 */
	void Sigmoid();
	/**
	 * @brief      set each element x to be log(x)
	 */
	void Log();
	/**
	 * @brief      set each element x to be exp(x)
	 */
	void Exp();

	/**
	 * @brief      truncate the values outside of the range
	 *
	 * @param[in]  lb    The lower bound
	 * @param[in]  ub    The upper bound
	 */
	void Truncate(Dtype lb, Dtype ub);
	
	/**
	 * @brief      print tensor to screen; for debug purpose
	 */
	void Print2Screen(); 
	
	/**
	 * the shared ptr to the data structure (which is used to keep the data of this tensor)
	 */
	std::shared_ptr< DenseData<CPU, Dtype> > data;
};

/**
 * @brief      CPU DENSE int tensor specialization; this tensor is not used for heavy computation
 * 				(e.g., matmul)
 */
template<>
class TensorTemplate<CPU, DENSE, int> : public Tensor
{
public:

	TensorTemplate();
	virtual ~TensorTemplate() {}
	
	virtual void Reshape(std::vector<size_t> l) override;
	virtual MatType GetMatType() override;
	virtual MatMode GetMatMode() override;

	/**
	 * @brief      save to disk
	 *
	 * @param      fid   The file handle
	 */
	virtual void Serialize(FILE* fid) override;

	/**
	 * @brief      load from disk
	 *
	 * @param      fid   The file handle
	 */
	virtual void Deserialize(FILE* fid) override;

	/**
	 * @brief      deeply copy src to this tensor
	 *
	 * @param      src   the CPU dense tensor with same data type
	 */
	void CopyFrom(DTensor<CPU, int>& src);		
#ifdef USE_GPU	
	/**
	 * @brief      deeply copy src to this tensor
	 *
	 * @param      src   the GPU dense tensor with same data type
	 */
	void CopyFrom(DTensor<GPU, int>& src);
#endif
	/**
	 * @brief      shallow copy (only set the shared_ptr)
	 *
	 * @param      src   The source
	 */
	void ShallowCopy(DTensor<CPU, int>& src);

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
	std::shared_ptr< DenseData<CPU, int> > data;
};

}
#endif