#include "nn/axpby.h"

namespace gnn
{

template<typename mode, typename Dtype>
Axpby<mode, Dtype>::Axpby(std::string _name, Dtype _a, Dtype _b, PropErr _properr) 
		: Factor(_name, _properr), a(_a), b(_b)
{

}

template<typename mode, typename Dtype>
void Axpby<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& x = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& y = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;

	output.CopyFrom(y);
	output.Axpby(a, x, b);
}

template<typename mode, typename Dtype>
void Axpby<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

	for (size_t i = 0; i < operands.size(); ++i)
	{
		if (isConst[i])
			continue;
		auto& grad_i = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->grad;		
		ASSERT(grad_i.shape == cur_grad.shape, "no broadcasting is supported right now");		
		grad_i.Full().Axpy( i == 0 ? a : b, cur_grad);
	}
}

INSTANTIATE_CLASS(Axpby)

}