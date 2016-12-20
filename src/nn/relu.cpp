#include "nn/relu.h"

namespace gnn
{

template<typename Dtype>
void ReLUAct(DTensor<CPU, Dtype>& in, DTensor<CPU, Dtype>& out)
{
	out.CopyFrom(in);
	for (size_t i = 0; i < out.shape.Count(); ++i)
		if (out.data->ptr[i] < 0)
			out.data->ptr[i] = 0;
}

template<typename mode, typename Dtype>
ReLU<mode, Dtype>::ReLU(std::string _name, PropErr _properr) 
					: Factor(_name, _properr)
{
}

template<typename mode, typename Dtype>
void ReLU<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 		std::vector< std::shared_ptr<Variable> >& outputs) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	ReLUAct(input, output);
}

template class ReLU<CPU, float>;
template class ReLU<CPU, double>;

}