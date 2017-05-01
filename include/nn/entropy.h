#ifndef ENTROPY_H
#define ENTROPY_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

template<typename Dtype>
void CalcEntropy(DTensor<CPU, Dtype>& prob, DTensor<CPU, Dtype>& out);

template<typename Dtype>
void CalcEntropy(DTensor<GPU, Dtype>& prob, DTensor<GPU, Dtype>& out);

/**
 * @brief      Operator for calculating entropy
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class Entropy : public Factor
{
public:
	static std::string StrType()
	{
		return "Entropy";
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
	Entropy(std::string _name, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< std::shared_ptr<Variable> >& outputs, 
						Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;
};

}

#endif