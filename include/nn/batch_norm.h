#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

template<typename mode, typename Dtype>
class BatchNorm : public Factor
{
public:
	static std::string StrType()
	{
		return "BatchNorm";
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

	BatchNorm(std::string _name, Dtype _alpha, PropErr _properr = PropErr::T);

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
};

}

#endif