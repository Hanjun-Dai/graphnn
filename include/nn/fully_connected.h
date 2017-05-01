#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

/**
 * @brief      fully connected operator
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { the input and output data type (float/double) }
 */
template<typename mode, typename Dtype>
class FullyConnected : public Factor
{
public:
	static std::string StrType()
	{
		return "FullyConnected";
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
	 * @param[in]  _properr  Whether propagate error
	 */
	FullyConnected(std::string _name, PropErr _properr = PropErr::T);
	
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< std::shared_ptr<Variable> >& outputs, 
						Phase phase) override;
	
	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;
};

}

#endif