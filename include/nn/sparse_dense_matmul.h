#ifndef SPARSE_DENSE_MATMUL_H
#define SPARSE_DENSE_MATMUL_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

/**
 * @brief      sparse dense matrix multiplication operator, optimized for gradient computation
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { the input and output data type (float/double) }
 */
template<typename mode, typename Dtype>
class SparseDenseMatMul : public Factor
{
public:
	static std::string StrType()
	{
		return "SparseDenseMatMul";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     return a matrix with the same data type as inputs
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name     The name
	 * @param[in]  _transA   Whether to transpose operand A
	 * @param[in]  _transB   Whether to transpose operand B
	 * @param[in]  _properr  Whether propagate error
	 */
	SparseDenseMatMul(std::string _name, Trans _transA = Trans::N, 
			Trans _transB = Trans::N, PropErr _properr = PropErr::T);
	
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< std::shared_ptr<Variable> >& outputs, 
						Phase phase) override;
	
	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;

	/**
	 * whether to transpose operand A
	 */
	Trans transA;

	/**
	 * Whether to transpose operand B
	 */
	Trans transB;
};

}

#endif