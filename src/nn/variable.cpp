#include "nn/variable.h"

namespace gnn
{

Variable::Variable(std::string _name) : name(_name), g(nullptr)
{

}

template<typename mode, typename Dtype>
TensorVarTemplate<mode, DENSE, Dtype>::TensorVarTemplate(std::string _name) : TensorVar<mode, Dtype>(_name)
{

}

template<typename mode, typename Dtype>
TensorVarTemplate<mode, DENSE, Dtype>::TensorVarTemplate(std::string _name, std::vector<size_t> l)
				 : TensorVar<mode, Dtype>(_name)
{
	value.Reshape(l);
}

template<typename mode, typename Dtype>
void TensorVarTemplate<mode, DENSE, Dtype>::SetRef(void* p)
{
	auto pt = static_cast< DTensor<mode, Dtype>* >(p);
	this->value.ShallowCopy(*pt);
}

template<typename mode, typename Dtype>
Dtype TensorVarTemplate<mode, DENSE, Dtype>::AsScalar()
{
	return value.AsScalar();
}

template<typename mode, typename Dtype>
MatType TensorVarTemplate<mode, DENSE, Dtype>::GetMatType()
{
	return MatType::dense;
}

template class TensorVarTemplate<CPU, DENSE, float>;
template class TensorVarTemplate<CPU, DENSE, double>;
template class TensorVarTemplate<CPU, DENSE, int>;

//============ SPARSE Tensor Variable ==================

template<typename mode, typename Dtype>
TensorVarTemplate<mode, SPARSE, Dtype>::TensorVarTemplate(std::string _name) : TensorVar<mode, Dtype>(_name)
{

}

template<typename mode, typename Dtype>
void TensorVarTemplate<mode, SPARSE, Dtype>::SetRef(void* p)
{
	auto* pt = static_cast< SpTensor<mode, Dtype>* >(p);
	this->value.ShallowCopy(*pt);
}

template<typename mode, typename Dtype>
Dtype TensorVarTemplate<mode, SPARSE, Dtype>::AsScalar()
{
	return 0;
}

template<typename mode, typename Dtype>
MatType TensorVarTemplate<mode, SPARSE, Dtype>::GetMatType()
{
	return MatType::sparse;
}

template class TensorVarTemplate<CPU, SPARSE, float>;
template class TensorVarTemplate<CPU, SPARSE, double>;
template class TensorVarTemplate<CPU, SPARSE, int>;

}