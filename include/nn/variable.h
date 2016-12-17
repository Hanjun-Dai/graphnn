#ifndef VARIABLE_H
#define VARIABLE_H

#include <string>
#include <vector>
#include "tensor/tensor.h"
#include "tensor/dense_tensor.h"

namespace gnn
{

class FactorGraph;

class Variable
{
public:
	Variable(std::string _name);

	std::string name;
	FactorGraph* g;
};

class ConstVar : public Variable
{
public:
	ConstVar(std::string _name);
};

class DiffVar : public Variable
{
public:
	DiffVar(std::string _name);
};

template<typename mode, typename Dtype>
class DTensorVar : public DiffVar
{
public: 
	DTensorVar(std::string _name);
	DTensorVar(std::string _name, std::initializer_list<uint> l);

	DTensor< mode, Dtype > value, grad;
};

template<typename mode, typename Dtype>
class SpTensorVar : public ConstVar
{
public:
	SpTensorVar(std::string _name);
};

}
#endif