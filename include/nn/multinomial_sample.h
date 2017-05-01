#ifndef MULTINOMIAL_SAMPLE_H
#define MULTINOMIAL_SAMPLE_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"
#include <random>

namespace gnn
{

/**
 * @brief      Class for multinomial sample.
 *
 * @tparam     mode  cpu/gpu
 * @tparam     Dtype    int/float/double
 */
template<typename mode, typename Dtype>
class MultinomialSample : public Factor
{
public:
	static std::string StrType()
	{
		return "MultinomialSample";
	}

	using OutType = std::shared_ptr< TensorVarTemplate<CPU, DENSE, int> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     sampled indexes, organized in either dense/sparse matrix
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< TensorVarTemplate<CPU, DENSE, int> >(out_name);
	}

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name     The name
	 * @param[in]  _need_softmax  Whether need to do softmax
	 */
	MultinomialSample(std::string _name, bool _need_softmax);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;
	/**
	 * whether need to do softmax for the input (whether the input is logits)
	 */
	bool need_softmax;	

	std::default_random_engine generator;
  	std::uniform_real_distribution<Dtype>* distribution;
};

}
#endif