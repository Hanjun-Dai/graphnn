#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

template<typename Dtype>
void CalcCrossEntropy(DTensor<CPU, Dtype>& prob, SpTensor<CPU, Dtype>& label, DTensor<CPU, Dtype>& out);

template<typename Dtype>
void CalcCrossEntropy(DTensor<GPU, Dtype>& prob, SpTensor<GPU, Dtype>& label, DTensor<GPU, Dtype>& out);

/**
 * @brief      Operator for cross entropy
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class CrossEntropy : public Factor
{
public:
	static std::string StrType()
	{
		return "CrossEntropy";
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
	 * @param[in]  _need_softmax  Whether need to do softmax before calculating cross entropy
	 * @param[in]  _properr       Whether propagete error
	 */
	CrossEntropy(std::string _name, bool _need_softmax, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< std::shared_ptr<Variable> >& outputs, 
						Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;

	/**
	 * whether need to do softmax for the input (whether the input is logits)
	 */
	bool need_softmax;

	/**
	 * temporary variable used for calculating probability
	 */
	DTensor<mode, Dtype> probs;
};

}

#endif