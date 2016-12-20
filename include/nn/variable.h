#ifndef VARIABLE_H
#define VARIABLE_H

#include <string>
#include <vector>
#include "tensor/tensor.h"
#include "tensor/dense_tensor.h"
#include "tensor/sparse_tensor.h"

namespace gnn
{

class FactorGraph;

class Variable
{
public:
	Variable(std::string _name);
	virtual void SetRef(void* p) NOT_IMPLEMENTED

	virtual EleType GetEleType() = 0;
	std::string name;
	FactorGraph* g;
};

template<typename mode, typename matType, typename Dtype>
class TensorVarTemplate;

template<typename mode, typename Dtype>
using DTensorVar = TensorVarTemplate<mode, DENSE, Dtype>;

template<typename mode, typename Dtype>
using SpTensorVar = TensorVarTemplate<mode, SPARSE, Dtype>;

template<typename mode, typename Dtype>
class TensorVar : public Variable
{
public:
	template<typename matType>
	TensorVarTemplate<mode, matType, Dtype>& Derived()
	{
		auto* ret = dynamic_cast<TensorVarTemplate<mode, matType, Dtype>*>(this);
		ASSERT(ret, "wrong derived type of TensorVar");
		return *ret;
	}

	virtual EleType GetEleType() override
	{
		return Dtype2Enum<Dtype>();
	}

	TensorVar(std::string _name) : Variable(_name) {}
	virtual Dtype AsScalar() = 0;
	virtual MatType GetMatType() = 0;
};

template<typename mode, typename matType, typename Dtype>
class TensorVarTemplate : TensorVar<mode, Dtype> {};

template<typename mode, typename Dtype>
class TensorVarTemplate<mode, DENSE, Dtype> : public TensorVar<mode, Dtype>
{
public: 
	TensorVarTemplate(std::string _name);
	TensorVarTemplate(std::string _name, std::vector<size_t> l);

	TensorVarTemplate(std::string _name, std::vector<int> l)
		: TensorVarTemplate(_name, std::vector<size_t>(l.begin(), l.end())) {}

	TensorVarTemplate(std::string _name, std::vector<uint> l)
		: TensorVarTemplate(_name, std::vector<size_t>(l.begin(), l.end())) {}

	virtual void SetRef(void* p) override;
	virtual Dtype AsScalar() override;
	virtual MatType GetMatType() override;
	DTensor< mode, Dtype > value, grad;
};

template<typename mode, typename Dtype>
class TensorVarTemplate<mode, SPARSE, Dtype> : public TensorVar<mode, Dtype>
{
public:
	TensorVarTemplate(std::string _name);
	virtual void SetRef(void* p) override;
	virtual Dtype AsScalar() override;
	virtual MatType GetMatType() override;

	SpTensor<mode, Dtype> value;
};

}
#endif