#include "nn/elewise_minus.h"

namespace gnn
{
template<typename mode, typename Dtype>
ElewiseMinus<mode, Dtype>::ElewiseMinus(std::string _name, PropErr _properr) 
		: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void ElewiseMinus<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& lhs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& rhs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;

	output.CopyFrom(lhs);
	output.Axpy(-1.0, rhs);	
}

template<typename mode, typename Dtype>
void ElewiseMinus<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
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
		auto grad_i = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->grad.Full();
		grad_i.Axpy((i == 0) ? 1.0 : -1.0, cur_grad);
	}
}

INSTANTIATE_CLASS(ElewiseMinus)

}