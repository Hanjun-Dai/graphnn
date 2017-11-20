#ifndef FACTOR_H
#define FACTOR_H

#include "nn/variable.h"
#include <string>
#include "util/fmt.h"

namespace gnn
{

/**
 * @brief      Abstract class of operators. Since we represent the computation graph as a 
 *             factor graph, the factors here represent the relations between variables
 */
class Factor
{
public:
	/**
	 * @brief      Constructor
	 *
	 * @param[in]  _name     The name of this factor
	 * @param[in]  _properr  Whether propagate error
	 */
	Factor(std::string _name, PropErr _properr = PropErr::T) : name(_name), properr(_properr) {}

	/**
	 * @brief      Forward function
	 *
	 * @param      operands  The input arguments (variables) to this operator
	 * @param      outputs   The output variables produced by this operator
	 * @param      phase     train/test
	 */
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) NOT_IMPLEMENTED

	/**
	 * @brief      Backward function
	 *
	 * @param      operands  The input arguments (variables) to this operator
	 * @param      outputs   The output variables produced by this operator
	 */
	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						  std::vector< bool >& isConst, 
						  std::vector< std::shared_ptr<Variable> >& outputs) NOT_IMPLEMENTED

	/**
	 * the name (identifier) of this operator
	 */
	std::string name;

	/**
	 * whether propagate error in backward stage
	 */
	PropErr properr;
};

}

#endif