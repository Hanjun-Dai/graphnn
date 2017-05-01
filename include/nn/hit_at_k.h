#ifndef HIT_AT_K_H
#define HIT_AT_K_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

template<typename Dtype>
void HitsTopK(DTensor<CPU, Dtype>& pred, SpTensor<CPU, Dtype>& label, DTensor<CPU, int>& out, int k);

template<typename Dtype>
void HitsTopK(DTensor<GPU, Dtype>& pred, SpTensor<GPU, Dtype>& label, DTensor<GPU, int>& out, int k);

/**
 * @brief      Operator: whether the top-k predictions hit the label set
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { prediction ele_type (float/double) }
 */
template<typename mode, typename Dtype>
class HitAtK : public Factor
{
public:
	static std::string StrType()
	{
		return "HitAtK";
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
	HitAtK(std::string _name, int _topK = 1);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	/**
	 * top K (by default 1)
	 */
	int topK;
};

}

#endif