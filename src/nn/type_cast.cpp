#include "nn/type_cast.h"

namespace gnn
{

template<typename mode, typename Dtype>
TypeCast<mode, Dtype>::TypeCast(std::string _name, PropErr _properr) 
				: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void TypeCast<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 


	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	ELE_TYPE_SWITCH(operands[0]->GetEleType(), eleType, {
		auto& input_var = dynamic_cast<DTensorVar<mode, eleType>*>(operands[0].get())->value;
		output.CopyFrom(input_var);
	});
	
}

INSTANTIATE_CLASS(TypeCast)

}