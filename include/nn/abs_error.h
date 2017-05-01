#ifndef ABS_ERROR_H
#define ABS_ERROR_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

/**
 * @brief      Operator for abs error
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class AbsError : public Factor
{
public:
	static std::string StrType()
	{
		return "AbsError";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     { return a vector with same Dtype as prediction }
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name          The name
	 * @param[in]  _properr       Whether propagete error
	 */
	AbsError(std::string _name, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< std::shared_ptr<Variable> >& outputs, 
						Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;

	DTensor<mode, Dtype> diff;
};

}

#endif