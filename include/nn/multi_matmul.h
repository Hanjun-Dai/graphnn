#ifndef MULTI_MATMUL_H
#define MULTI_MATMUL_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

/**
 * @brief      multiple pairs of matrix multiplication operator; this is used 
 * 				to save space (i.e., instead of multiple matmul + elewise-add layer)
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { the input and output data type (float/double) }
 */
template<typename mode, typename Dtype>
class MultiMatMul : public Factor
{
public:
	static std::string StrType()
	{
		return "MultiMatMul";
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
	 * @brief      constructor; no transpose is supported right now
	 *
	 * @param[in]  _name     The name
	 * @param[in]  _properr  Whether propagate error
	 */
	MultiMatMul(std::string _name, PropErr _properr = PropErr::T);
	
	/**
	 * @brief      forward function
	 *
	 * @param      operands  The operands, in the form of: op[0] * op[1] + op[2] * op[3] + ...
	 * @param      outputs   The single output 
	 * @param      phase     train/test
	 */
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< std::shared_ptr<Variable> >& outputs, 
						Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;
};

}

#endif