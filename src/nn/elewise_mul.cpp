#include "nn/elewise_mul.h"

namespace gnn
{
template<typename mode, typename Dtype>
ElewiseMul<mode, Dtype>::ElewiseMul(std::string _name, PropErr _properr) 
		: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void ElewiseMul<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() >= 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& op1 = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	output.CopyFrom(op1);
	for (size_t i = 1; i < operands.size(); ++i)
	{
		auto& op_i = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->value;
		ASSERT(op_i.shape == output.shape, "no broadcasting is supported right now");
		output.ElewiseMul(op_i);
	}
}

template<typename mode, typename Dtype>
void ElewiseMul<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
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
		
		DTensor<mode, Dtype> prev_grad;
		prev_grad.CopyFrom(cur_grad);
		
		for (size_t j = 0; j < operands.size(); ++j)	
			if (j != i)
			{
				auto& state_j = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[j].get())->value;
				prev_grad.ElewiseMul(state_j);
			}

		auto grad_i = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->grad.Full();
		ASSERT(grad_i.shape == cur_grad.shape, "no broadcasting is supported right now");			
		grad_i.Axpy(1.0, prev_grad);
	}
}

INSTANTIATE_CLASS(ElewiseMul)

}