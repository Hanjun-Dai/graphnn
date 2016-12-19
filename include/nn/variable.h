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

template<typename mode, typename Dtype>
class TensorVar : public Variable
{
public:
	TensorVar(std::string _name) : Variable(_name) {}
	virtual Dtype AsScalar() = 0;
};

template<typename mode, typename Dtype>
class DTensorVar : public TensorVar<mode, Dtype>
{
public: 
	DTensorVar(std::string _name);
	DTensorVar(std::string _name, std::initializer_list<uint> l);

	virtual Dtype AsScalar() override;
	DTensor< mode, Dtype > value, grad;
};

template<typename mode, typename Dtype>
class SpTensorVar : public TensorVar<mode, Dtype>
{
public:
	SpTensorVar(std::string _name);

	virtual Dtype AsScalar() override;
};

}
#endif