#include "nn/square_error.h"

namespace gnn
{

template<typename mode, typename Dtype>
SquareError<mode, Dtype>::SquareError(std::string _name, PropErr _properr) 
				: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void SquareError<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 				std::vector< std::shared_ptr<Variable> >& outputs, 
						 				Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& pred = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& label = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;
	
	diff.CopyFrom(pred);
	diff.Axpy(-1.0, label);

	output.CopyFrom(diff);
	output.Square();
}

template<typename mode, typename Dtype>
void SquareError<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
										std::vector< bool >& isConst, 
						 				std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType());
	if (isConst[0])
		return;
	auto grad_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();
	auto grad_lhs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();
	
	diff.ElewiseMul(grad_out);
	grad_lhs.Axpy(2.0, diff);
}

INSTANTIATE_CLASS(SquareError)

}