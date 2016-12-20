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
using SpTensor = TensorTemplate<mode, SPARSE, Dtype>;

/**
 * @brief      Abstract Class for tensor.
 */
class Tensor
{
public:
	template<typename mode, typename matType, typename Dtype>
	TensorTemplate<mode, matType, Dtype>& Derived();

	uint rank();
	size_t rows();
	size_t cols();

	virtual void Reshape(std::vector<size_t> l) NOT_IMPLEMENTED

	virtual MatType GetMatType() = 0;

	virtual MatMode GetMatMode() = 0;

	/**
	 * Tensor shape
	 */
	TShape shape;
private:

};

template<typename mode, typename matType, typename Dtype>
class TensorTemplate : public Tensor {};


} // namespace gnn
#endif