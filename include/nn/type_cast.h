#ifndef TYPE_CAST_H
#define TYPE_CAST_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

/**
 * @brief      operator used for casting type
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { output type, float/double/int }
 */
template<typename mode, typename Dtype>
class TypeCast : public Factor
{
public:
	static std::string StrType()
	{
		return "TypeCast";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     return a tensor with same shape as input, but the output 
	 * 				tensor has the data type specified by Dtype, which is 
	 * 				independent of the data type from input tensor
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
	 * @param[in]  _properr  whethre propagate error
	 */
	TypeCast(std::string _name, PropErr _properr = PropErr::T);
	
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

};

}


#endif