#ifndef IN_TOP_H_H
#define IN_TOP_H_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

template<typename Dtype>
void IsInTopK(DTensor<CPU, Dtype>& pred, DTensor<CPU, int>& label, DTensor<CPU, int>& out, int k);

template<typename Dtype>
void IsInTopK(DTensor<GPU, Dtype>& pred, DTensor<GPU, int>& label, DTensor<GPU, int>& out, int k);

/**
 * @brief      Operator: whether the true label is in top-k of prediction
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { prediction ele_type (float/double) }
 */
template<typename mode, typename Dtype>
class InTopK : public Factor
{
public:
	static std::string StrType()
	{
		return "InTopK";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, int> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     an integer tensor (a 0/1 vector)
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
	 * @param[in]  _topK  The top K
	 */
	InTopK(std::string _name, int _topK = 1);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	/**
	 * top K (by default 1, which corresponds to accuracy)
	 */
	int topK;
};

}

#endif