#include "nn/arg_max.h"

namespace gnn
{


template<typename mode, typename Dtype>
ArgMax<mode, Dtype>::ArgMax(std::string _name, uint _axis) 
				: Factor(_name, PropErr::N), axis(_axis)
{
	ASSERT(axis == 0, StrType() << " not implemented for axis other than 0");
}

template<typename mode, typename Dtype>
void ArgMax<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs,
						 			Phase phase)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, int>*>(outputs[0].get())->value;
	auto* input_var = dynamic_cast< TensorVar<mode, Dtype>* >(operands[0].get());
	
	MAT_TYPE_SWITCH(input_var->GetMatType(), matType, {	
		auto& input = input_var->template Derived<matType>().value;
		input.ArgMax(output);
	});
}

INSTANTIATE_CLASS(ArgMax)

}