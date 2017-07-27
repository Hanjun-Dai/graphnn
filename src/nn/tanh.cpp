#include "nn/tanh.h"

namespace gnn
{

template<typename Dtype>
void TanhAct(DTensor<CPU, Dtype>& in, DTensor<CPU, Dtype>& out)
{
	out.Reshape(in.shape.dims);
    Dtype x, y;
    auto cnt = in.shape.Count();
    for (size_t i = 0; i < cnt; ++i)
    {
        x = exp(in.data->ptr[i]); 
        y = exp(-in.data->ptr[i]);
        out.data->ptr[i] = (x - y) / (x + y);
    }
}

template<typename Dtype>
void TanhDeriv(DTensor<CPU, Dtype>& dst, DTensor<CPU, Dtype>& cur_out, DTensor<CPU, Dtype>& cur_grad)
{
	auto cnt = dst.shape.Count();
	for (size_t i = 0; i < cnt; ++i)
		dst.data->ptr[i] += cur_grad.data->ptr[i] * (1 - cur_out.data->ptr[i] * cur_out.data->ptr[i]);
}

template<typename mode, typename Dtype>
Tanh<mode, Dtype>::Tanh(std::string _name, PropErr _properr) 
					: Factor(_name, _properr)
{
}

template<typename mode, typename Dtype>
void Tanh<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 		std::vector< std::shared_ptr<Variable> >& outputs, 
						 		Phase phase) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	TanhAct(input, output);
}

template<typename mode, typename Dtype>
void Tanh<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
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

	TanhDeriv(prev_grad, cur_out, cur_grad);
}

INSTANTIATE_CLASS(Tanh)

}