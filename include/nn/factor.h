#ifndef FACTOR_H
#define FACTOR_H

#include <string>

namespace gnn
{

class Factor
{
public:
	Factor(std::string _name) : name(_name){}

	std::string name;
};

}

#endif