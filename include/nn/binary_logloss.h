#ifndef BINARY_LOGLOSS_H
#define BINARY_LOGLOSS_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

template<typename Dtype>
void CalcBinaryLogLoss(DTensor<CPU, Dtype>& prob, DTensor<CPU, Dtype>& label, DTensor<CPU, Dtype>& out);

template<typename Dtype>
void CalcBinaryLogLoss(DTensor<GPU, Dtype>& prob, DTensor<GPU, Dtype>& label, DTensor<GPU, Dtype>& out);

/**
 * @brief      Operator for binary log loss
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class BinaryLogLoss : public Factor
{
public:
	static std::string StrType()
	{
		return "BinaryLogLoss";
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
	 * @param[in]  _need_sigmoid  Whether need sigmoid before calculating BinaryLogLoss
	 * @param[in]  _properr       Whether propagete error
	 */
	BinaryLogLoss(std::string _name, bool _need_sigmoid, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< std::shared_ptr<Variable> >& outputs, 
						Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;

	/**
	 * whether need to do sigmoid for the input (whether the input is not prob)
	 */
	bool need_sigmoid;
	/**
	 * temporary variable used for calculating probability
	 */
	DTensor<mode, Dtype> tmp_probs;
};

}

#endif