#include "nn/identity.h"

namespace gnn
{
template<typename mode, typename Dtype>
Identity<mode, Dtype>::Identity(std::string _name, PropErr _properr) 
		: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void Identity<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	MAT_MODE_SWITCH(operands[0]->GetMode(), matMode, {
		auto& input = dynamic_cast<DTensorVar<matMode, Dtype>*>(operands[0].get())->value;
		output.CopyFrom(input);	
	});
}

template<typename mode, typename Dtype>
void Identity<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

	if (!isConst[0])
	{	
		MAT_MODE_SWITCH(operands[0]->GetMode(), matMode, {
			auto prev_grad = dynamic_cast<DTensorVar<matMode, Dtype>*>(operands[0].get())->grad.Full();
			DTensor<matMode, Dtype> buf;
			buf.CopyFrom(cur_grad);			
			prev_grad.Axpy(1.0, buf);
		});	
	}	
}

INSTANTIATE_CLASS(Identity)

}