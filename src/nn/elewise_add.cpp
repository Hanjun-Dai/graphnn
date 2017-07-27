#include "nn/elewise_add.h"

namespace gnn
{
template<typename mode, typename Dtype>
ElewiseAdd<mode, Dtype>::ElewiseAdd(std::string _name, std::vector<Dtype> _coeff, PropErr _properr) 
		: Factor(_name, _properr), coeff(_coeff)
{

}

template<typename mode, typename Dtype>
void ElewiseAdd<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() >= 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType());
	ASSERT(coeff.size() == 0 || operands.size() == coeff.size(), "wrong number of coefficients");

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& op1 = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	output.CopyFrom(op1);
	if (coeff.size())
		output.Scale(coeff[0]);
	for (size_t i = 1; i < operands.size(); ++i)
	{
		auto& op_i = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->value;
		ASSERT(op_i.shape == output.shape, "no broadcasting is supported right now");
		output.Axpy(coeff.size() ? coeff[i] : 1.0, op_i);
	}
}

template<typename mode, typename Dtype>
void ElewiseAdd<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() >= 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();
	for (size_t i = 0; i < operands.size(); ++i)
	{
		if (isConst[i])
			continue;
		auto grad_i = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->grad.Full();
		ASSERT(grad_i.shape == cur_grad.shape, "no broadcasting is supported right now");
		grad_i.Axpy(coeff.size() ? coeff[i] : 1.0, cur_grad);
	}
}

INSTANTIATE_CLASS(ElewiseAdd)

}