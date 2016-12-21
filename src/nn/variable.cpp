#include "nn/variable.h"

namespace gnn
{

Variable::Variable(std::string _name) : name(_name)
{

}

bool Variable::IsConst()
{
	return true;
}

template<typename mode, typename Dtype>
TensorVarTemplate<mode, DENSE, Dtype>::TensorVarTemplate(std::string _name, bool _isConst)
			 : TensorVar<mode, Dtype>(_name), isConst(_isConst)
{
	if (Dtype2Enum<Dtype>() == EleType::INT32)
		isConst = true;
}

template<typename mode, typename Dtype>
TensorVarTemplate<mode, DENSE, Dtype>::TensorVarTemplate(std::string _name, std::vector<size_t> l, bool _isConst)
				 : TensorVar<mode, Dtype>(_name), isConst(_isConst)
{
	if (Dtype2Enum<Dtype>() == EleType::INT32)
		isConst = true;
	value.Reshape(l);
}

template<typename mode, typename Dtype>
void TensorVarTemplate<mode, DENSE, Dtype>::SetRef(void* p)
{
	auto pt = static_cast< DTensor<mode, Dtype>* >(p);
	this->value.ShallowCopy(*pt);
}

template<typename mode, typename Dtype>
bool TensorVarTemplate<mode, DENSE, Dtype>::IsConst()
{
	return this->isConst;
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

template<typename mode, typename Dtype>
void TensorVarTemplate<mode, DENSE, Dtype>::ZeroGrad()
{
	ASSERT(!isConst, "cannot set gradient for the constant variable");
	grad.Reshape(value.shape.dims);
	grad.Zeros();
}

template<typename mode, typename Dtype>
void TensorVarTemplate<mode, DENSE, Dtype>::OnesGrad()
{
	ASSERT(!isConst, "cannot set gradient for the constant variable");
	grad.Reshape(value.shape.dims);
	grad.Fill(1);
}

template class TensorVarTemplate<CPU, DENSE, float>;
template class TensorVarTemplate<CPU, DENSE, double>;
template class TensorVarTemplate<CPU, DENSE, int>;

//============ SPARSE Tensor Variable ==================

template<typename mode, typename Dtype>
TensorVarTemplate<mode, SPARSE, Dtype>::TensorVarTemplate(std::string _name) 
			: TensorVar<mode, Dtype>(_name)
{

}

template<typename mode, typename Dtype>
void TensorVarTemplate<mode, SPARSE, Dtype>::SetRef(void* p)
{
	auto* pt = static_cast< SpTensor<mode, Dtype>* >(p);
	this->value.ShallowCopy(*pt);
}

template<typename mode, typename Dtype>
bool TensorVarTemplate<mode, SPARSE, Dtype>::IsConst()
{
	return true;
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