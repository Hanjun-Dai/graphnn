#ifndef FACTOR_H
#define FACTOR_H

#include "nn/variable.h"
#include <string>

namespace gnn
{

class Factor
{
public:
	Factor(std::string _name, PropErr _properr = PropErr::T) : name(_name), properr(_properr) {}

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs) NOT_IMPLEMENTED

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						  std::vector< std::shared_ptr<Variable> >& outputs) NOT_IMPLEMENTED

	std::string name;
	PropErr properr;
};

}

#endif