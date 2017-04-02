#include "nn/kxplusb.h"

namespace gnn
{

template<typename mode, typename Dtype>
Kxplusb<mode, Dtype>::Kxplusb(std::string _name, Dtype _a, Dtype _b, PropErr _properr) 
		: Factor(_name, _properr), a(_a), b(_b)
{

}

template<typename mode, typename Dtype>
void Kxplusb<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& y = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& x = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	y.CopyFrom(x);
	y.Scale(a);
	y.Add(b);
}

template<typename mode, typename Dtype>
void Kxplusb<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad;

	for (size_t i = 0; i < operands.size(); ++i)
	{
		if (isConst[i])
			continue;
		auto& grad_i = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->grad;		
		ASSERT(grad_i.shape == cur_grad.shape, "no broadcasting is supported right now");		
		grad_i.Axpy( i == 0 ? a : b, cur_grad);
	}
}

template class Kxplusb<CPU, float>;
template class Kxplusb<CPU, double>;
template class Kxplusb<GPU, float>;
template class Kxplusb<GPU, double>;

}