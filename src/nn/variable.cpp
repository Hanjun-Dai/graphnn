#include "nn/variable.h"

namespace gnn
{

Variable::Variable(std::string _name) : name(_name)
{

}

ConstVar::ConstVar(std::string _name) : Variable(_name)
{
	
}

DiffVar::DiffVar(std::string _name) : Variable(_name)
{

}

template<typename mode, typename Dtype>
DTensorVar<mode, Dtype>::DTensorVar(std::string _name) : DiffVar(_name)
{

}

template<typename mode, typename Dtype>
DTensorVar<mode, Dtype>::DTensorVar(std::string _name, std::initializer_list<uint> l)
				 : DiffVar(_name)
{
	value.Reshape(l);
}

template<typename mode, typename Dtype>
SpTensorVar<mode, Dtype>::SpTensorVar(std::string _name) : ConstVar(_name)
{

}

template class DTensorVar<CPU, float>;
template class DTensorVar<CPU, double>;
template class DTensorVar<CPU, int>;

template class SpTensorVar<CPU, float>;
template class SpTensorVar<CPU, double>;
template class SpTensorVar<CPU, int>;

}