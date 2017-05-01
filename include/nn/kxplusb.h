#ifndef KXPLUSB_H
#define KXPLUSB_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

/**
 * @brief      Operator: y = k * x + b
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { ele_type (float/double) }
 */
template<typename mode, typename Dtype>
class Kxplusb : public Factor
{
public:
	static std::string StrType()
	{
		return "Kxplusb";
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
	 * @brief      Constructor
	 *
	 * @param[in]  _name     The name
	 * @param[in]  _k        coeff k
	 * @param[in]  _b        coeff b
	 * @param[in]  _properr  The properr
	 */
	Kxplusb(std::string _name, Dtype _k, Dtype _b, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;
	/**
	 * coeff k
	 */
	Dtype k;

	/**
	 * coeff b
	 */
	Dtype b;
};

}
#endif