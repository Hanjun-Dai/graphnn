#ifndef REDUCE_MEAN_H
#define REDUCE_MEAN_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

/**
 * @brief      The reduction operator for calculating mean value
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class ReduceMean : public Factor
{
public:
	static std::string StrType()
	{
		return "ReduceMean";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     return a tensor with same data type as input
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name      The name
	 * @param[in]  _axis      The axis to be reduce; the default value (-1) means reduce over entire tensor
	 * @param[in]  _keep_dim  Whether keep the rank of input tensor (replace the axis with 1 after reduction)
	 * @param[in]  _properr   whether propaget error
	 */
	ReduceMean(std::string _name, int _axis = -1, bool _keep_dim = false, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;
	
	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						  std::vector< bool >& isConst, 
						  std::vector< std::shared_ptr<Variable> >& outputs) override;

	/**
	 * the axis to be reduce
	 */
	int axis;
	/**
	 * whether keep the rank
	 */
	bool keep_dim;
};

}

#endif