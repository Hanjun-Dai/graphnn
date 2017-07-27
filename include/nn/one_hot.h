#ifndef ONE_HOT_H
#define ONE_HOT_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"
#include <random>

namespace gnn
{

/**
 * @brief      Class for one hot sparse representation
 *
 * @tparam     mode  cpu/gpu
 * @tparam     Dtype    float/double
 */
template<typename mode, typename Dtype>
class OneHot : public Factor
{
public:
	static std::string StrType()
	{
		return "OneHot";
	}

	using OutType = std::shared_ptr< TensorVarTemplate<mode, CSR_SPARSE, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     one hot sparse representation
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< TensorVarTemplate<mode, CSR_SPARSE, Dtype> >(out_name);
	}

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name     The name
	 * @param[in]  _dim     The dimension of output
	 */
	OneHot(std::string _name, size_t _dim);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	/**
	 * the output dimension
	 */
	size_t dim;
};

}
#endif