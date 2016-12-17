#ifndef FACTOR_H
#define FACTOR_H

#include <string>

namespace gnn
{

class Factor
{
public:
	Factor(std::string _name, PropErr _properr = PropErr::T) : name(_name), properr(_properr) {}

	std::string name;
	PropErr properr;
};

}

#endif