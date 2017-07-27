#ifndef TENSOR_H
#define TENSOR_H

#include "util/gnn_macros.h"
#include "t_shape.h"
#include <memory>
#include <iostream>
#include <exception>
#include <string>

namespace gnn {

#define GPU_T(x) (x == Trans::N ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T)
#define CUSP_T(x) (x == Trans::N ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE)
#define CPU_CharT(x) (x == Trans::N ? 'N' : 'T')
#define CPU_T(x) (x == Trans::N ? CblasNoTrans : CblasTrans)

inline void GetDims(const size_t& lhs_rows, const size_t& lhs_cols, Trans ltrans, 
					const size_t& rhs_rows, const size_t& rhs_cols, Trans rtrans, 
					size_t &m, size_t &n, size_t &k)
{
	m = ltrans == Trans::N ? lhs_rows : lhs_cols;
	n = rtrans == Trans::N ? rhs_cols : rhs_rows;
	k = ltrans == Trans::N ? lhs_cols : lhs_rows;
	ASSERT((rtrans == Trans::N && rhs_rows == k) || (rtrans == Trans::T && rhs_cols == k), "mat dim doesn't match");
}

template<typename mode, typename matType, typename Dtype>
class TensorTemplate;

template<typename mode, typename Dtype>
using DTensor = TensorTemplate<mode, DENSE, Dtype>;

template<typename mode, typename Dtype>
using SpTensor = TensorTemplate<mode, CSR_SPARSE, Dtype>;

template<typename mode, typename Dtype>
using RowSpTensor = TensorTemplate<mode, ROW_SPARSE, Dtype>;

/**
 * @brief      Abstract Class for tensor.
 */
class Tensor
{
public:
	virtual ~Tensor() {}
	/**
	 * @brief      get derived subclass from the pointer to this abstract class
	 *
	 * @tparam     mode     { CPU/GPU }
	 * @tparam     matType  { DENSE/CSR_SPARSE/ROW_SPARSE }
	 * @tparam     Dtype    { float/double/int }
	 *
	 * @return     the derived subclass
	 */
	template<typename mode, typename matType, typename Dtype>
	TensorTemplate<mode, matType, Dtype>& Derived();

	/**
	 * @brief      vector = rank 1, matrix = rank 2, etc
	 *
	 * @return     the rank
	 */
	inline uint rank()
	{
		return this->shape.dims.size();
	}
	/**
	 * @brief      # rows; only valid when rank <= 2
	 *
	 * @return     # rows;
	 */
	inline size_t rows()
	{
		ASSERT(rank() <= 2, "rows is not well defined for rank > 2");
		return this->shape[0];
	}

	/**
	 * @brief      # cols; only valid when rank <= 2
	 *
	 * @return     # cols
	 */
	inline size_t cols()
	{
		// assume column vector
		if (rank() == 1)
			return 1;
		if (rank() == 2)
			return this->shape[1];
		ASSERT(false, "cols is not well defined for rank > 2");
	}

	/**
	 * @brief      reshape the tensor
	 *
	 * @param[in]  l     a list specifying the new shape
	 */
	virtual void Reshape(std::vector<size_t> l) NOT_IMPLEMENTED

	/**
	 * @brief      Gets the matrix type.
	 *
	 * @return     DENSE/CSR_SPARSE/ROW_SPARSE enum
	 */
	virtual MatType GetMatType() = 0;

	/**
	 * @brief      Gets the matrix mode.
	 *
	 * @return     CPU/GPU enum
	 */
	virtual MatMode GetMatMode() = 0;
	
	/**
	 * @brief      save to disk
	 *
	 * @param      fid   The file handle
	 */
	virtual void Serialize(FILE* fid);

	/**
	 * @brief      load from disk
	 *
	 * @param      fid   The file handle
	 */
	virtual void Deserialize(FILE* fid);

	/**
	 * Tensor shape
	 */
	TShape shape;
private:

};

/**
 * @brief      the implementation of abstract tensor
 *
 * @tparam     mode     { CPU/GPU }	
 * @tparam     matType  { DENSE/CSR_SPARSE/ROW_SPARSE }
 * @tparam     Dtype    { float/double/int }
 */
template<typename mode, typename matType, typename Dtype>
class TensorTemplate : public Tensor {};


} // namespace gnn
#endif