#ifndef ELEWISE_MINUS_H
#define ELEWISE_MINUS_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

/**
 * @brief      Operator: element-wise minus of two tensors; broadcast support
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { ele_type (float/double) }
 */
template<typename mode, typename Dtype>
class ElewiseMinus : public Factor
{
public:
	static std::string StrType()
	{
		return "ElewiseMinus";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     a tensor with the same shape as inputs
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
	 * @param[in]  _properr  The properr
	 */
	ElewiseMinus(std::string _name, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;
};

}
#endif