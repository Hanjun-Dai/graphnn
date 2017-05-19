#ifndef MOVING_NORM_H
#define MOVING_NORM_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

template<typename mode, typename Dtype>
class MovingNorm : public Factor
{
public:
	static std::string StrType()
	{
		return "MovingNorm";
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

	MovingNorm(std::string _name, Dtype _alpha, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;

	/**
	 * smoothing factor; c = a * c_1 + (1-a) * c
	 */
	Dtype alpha;

	/**
	 * eps to prevent 'divide by zero'
	 */
	Dtype eps;

	/**
	 * truncate inv std
	 */
	DTensor<mode, Dtype> normed_inv_std;

	bool has_first;
};

}

#endif
