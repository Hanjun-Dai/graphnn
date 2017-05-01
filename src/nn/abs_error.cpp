#include "nn/abs_error.h"

namespace gnn
{

template<typename mode, typename Dtype>
AbsError<mode, Dtype>::AbsError(std::string _name, PropErr _properr) 
				: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void AbsError<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 				std::vector< std::shared_ptr<Variable> >& outputs, 
						 				Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& pred = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& label = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;
	
	output.CopyFrom(pred);
	output.Axpy(-1.0, label);

	output.Abs();
}

template<typename mode, typename Dtype>
void AbsError<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
										std::vector< bool >& isConst, 
						 				std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(false, "bp is not implemented yet in abs_error");
}

INSTANTIATE_CLASS(AbsError)

}