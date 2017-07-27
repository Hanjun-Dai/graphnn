#include "nn/relu.h"

namespace gnn
{

template<typename Dtype>
void ReLUAct(DTensor<CPU, Dtype>& in, DTensor<CPU, Dtype>& out)
{
	out.CopyFrom(in);
	auto cnt = out.shape.Count();
	for (size_t i = 0; i < cnt; ++i)
		if (out.data->ptr[i] < 0)
			out.data->ptr[i] = 0;
}

template<typename Dtype>
void ReLUDeriv(DTensor<CPU, Dtype>& dst, DTensor<CPU, Dtype>& cur_out, DTensor<CPU, Dtype>& cur_grad)
{
	auto cnt = dst.shape.Count();
	for (size_t i = 0; i < cnt; ++i)
		if (cur_out.data->ptr[i] > 0.0)
			dst.data->ptr[i] += cur_grad.data->ptr[i];
}

template<typename mode, typename Dtype>
ReLU<mode, Dtype>::ReLU(std::string _name, PropErr _properr) 
					: Factor(_name, _properr)
{
}

template<typename mode, typename Dtype>
void ReLU<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 		std::vector< std::shared_ptr<Variable> >& outputs, 
						 		Phase phase) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	ReLUAct(input, output);
}

template<typename mode, typename Dtype>
void ReLU<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
								std::vector< bool >& isConst, 
						 		std::vector< std::shared_ptr<Variable> >& outputs) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 
	if (isConst[0])
		return;
	auto* var_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get());
	auto& cur_out = var_out->value;
	auto cur_grad = var_out->grad.Full();

	auto prev_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();

	ReLUDeriv(prev_grad, cur_out, cur_grad);
}

INSTANTIATE_CLASS(ReLU)

}