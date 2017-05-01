#ifndef ARG_MAX_H
#define ARG_MAX_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

/**
 * @brief      Operator for argument maximum.
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double/int }
 */
template<typename mode, typename Dtype>
class ArgMax : public Factor
{
public:
	static std::string StrType()
	{
		return "ArgMax";
	}

	/**
	 * return an int vector
	 */
	using OutType = std::shared_ptr< DTensorVar<mode, int> >;
	
	/**
	 * @brief      Creates an out variable.
	 *             
	 * @return     { return an int vector }
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, int> >(out_name);
	}

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name  The name
	 * @param[in]  _axis  keep which axis?
	 */
	ArgMax(std::string _name, uint _axis = 0);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	/**
	 * the axis to be kept
	 */
	uint axis;
};

}


#endif