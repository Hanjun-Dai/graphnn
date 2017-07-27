#include "nn/nat_log.h"

namespace gnn
{

template<typename mode, typename Dtype>
NatLog<mode, Dtype>::NatLog(std::string _name, PropErr _properr) 
					: Factor(_name, _properr)
{
}

template<typename mode, typename Dtype>
void NatLog<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 		std::vector< std::shared_ptr<Variable> >& outputs, 
						 		Phase phase) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	output.CopyFrom(input);
	output.Log();
}

template<typename mode, typename Dtype>
void NatLog<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
								std::vector< bool >& isConst, 
						 		std::vector< std::shared_ptr<Variable> >& outputs) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 
	if (isConst[0])
		return;
	DTensor<mode, Dtype> buf;

	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

	buf.CopyFrom(input);
	buf.Inv();

	buf.ElewiseMul(cur_grad);

	auto prev_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();

	prev_grad.Axpy(1.0, buf);
}

INSTANTIATE_CLASS(NatLog)

}