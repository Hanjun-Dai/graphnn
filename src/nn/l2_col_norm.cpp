#include "nn/l2_col_norm.h"
#include "tensor/mkl_helper.h"

namespace gnn
{

#define sqr(x) ((x) * (x))

template<typename Dtype>
void L2ColNormFwd(DTensor<CPU, Dtype>& in, DTensor<CPU, Dtype>& out, DTensor<CPU, Dtype>& norm2, DTensor<CPU, Dtype>& len)
{	
	norm2.Reshape({in.shape[0], 1});
	len.Reshape(norm2.shape.dims);

	size_t d = in.shape.Count(1);
	Dtype* ptr = in.data->ptr;
	for (size_t i = 0; i < in.shape[0]; ++i)
	{
		len.data->ptr[i] = MKL_Norm2(d, ptr) + 1e-10;
		norm2.data->ptr[i] = sqr(len.data->ptr[i]);
		ptr += d;
	}

	out.CopyFrom(in);
	out.ElewiseDiv(len);
}

template<typename Dtype>
void L2ColNormGrad(DTensor<CPU, Dtype>& x, DTensor<CPU, Dtype>& prev_grad, DTensor<CPU, Dtype>& cur_grad, DTensor<CPU, Dtype>& norm2, DTensor<CPU, Dtype>& len, Dtype scale)
{
	DTensor<CPU, Dtype> tmp(x.shape.dims);
	tmp.CopyFrom(x);
	tmp.ElewiseMul(cur_grad);

	size_t d = x.shape.Count(1);
	for (size_t i = 0; i < tmp.shape[0]; ++i)
	{
		Dtype* t_ptr = tmp.data->ptr + d * i;
		Dtype s = 0;
		for (size_t j = 0; j < d; ++j)
			s += t_ptr[j];
		norm2.data->ptr[i] = s / norm2.data->ptr[i] / len.data->ptr[i];
	}

	tmp.CopyFrom(x);
	tmp.ElewiseMul(norm2);
	prev_grad.Axpy(-scale, tmp);

	tmp.CopyFrom(cur_grad);
	tmp.ElewiseDiv(len);
	prev_grad.Axpy(scale, tmp);
}

template<typename mode, typename Dtype>
L2ColNorm<mode, Dtype>::L2ColNorm(std::string _name, Dtype _scale, PropErr _properr) 
					: Factor(_name, _properr), scale(_scale)
{

}

template<typename mode, typename Dtype>
void L2ColNorm<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 		std::vector< std::shared_ptr<Variable> >& outputs, 
						 		Phase phase) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	L2ColNormFwd(input, output, norm2, len);
	output.Scale(scale);
}

template<typename mode, typename Dtype>
void L2ColNorm<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
								std::vector< bool >& isConst, 
						 		std::vector< std::shared_ptr<Variable> >& outputs) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 
	if (isConst[0])
		return;
	auto* var_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get());
	auto cur_grad = var_out->grad.Full();

	auto prev_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();
	auto& prev_out = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	L2ColNormGrad(prev_out, prev_grad, cur_grad, norm2, len, scale);
}


INSTANTIATE_CLASS(L2ColNorm)

}
