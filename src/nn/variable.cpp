#include "nn/variable.h"

namespace gnn
{

Variable::Variable(std::string _name) : name(_name), g(nullptr)
{

}

template<typename mode, typename Dtype>
DTensorVar<mode, Dtype>::DTensorVar(std::string _name) : TensorVar<mode, Dtype>(_name)
{

}

template<typename mode, typename Dtype>
DTensorVar<mode, Dtype>::DTensorVar(std::string _name, std::initializer_list<uint> l)
				 : TensorVar<mode, Dtype>(_name)
{
	value.Reshape(l);
}

template<typename mode, typename Dtype>
Dtype DTensorVar<mode, Dtype>::AsScalar()
{
	return value.AsScalar();
}

template<typename mode, typename Dtype>
SpTensorVar<mode, Dtype>::SpTensorVar(std::string _name) : TensorVar<mode, Dtype>(_name)
{

}

template<typename mode, typename Dtype>
Dtype SpTensorVar<mode, Dtype>::AsScalar()
{
	return 0;
}

template class DTensorVar<CPU, float>;
template class DTensorVar<CPU, double>;
template class DTensorVar<CPU, int>;

template class SpTensorVar<CPU, float>;
template class SpTensorVar<CPU, double>;
template class SpTensorVar<CPU, int>;

}