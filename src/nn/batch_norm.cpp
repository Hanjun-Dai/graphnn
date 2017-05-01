#include "nn/batch_norm.h"

namespace gnn
{

template<typename mode, typename Dtype>
BatchNorm<mode, Dtype>::BatchNorm(std::string _name, Dtype _alpha, PropErr _properr) 
		: Factor(_name, _properr), alpha(_alpha)
{

}

template<typename mode, typename Dtype>
void BatchNorm<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 3, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& moving_mean = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;
	auto& moving_std = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[2].get())->value;

	output.CopyFrom(input);
	auto mul = DTensor<mode, Dtype>({input.rows(), (size_t)1});
	mul.Fill(1.0);
	if (phase == Phase::TRAIN)
	{
		DTensor<mode, Dtype> cur_mean;
		cur_mean.MM(mul, input, Trans::T, Trans::N, 1.0 / input.rows(), 0.0);

		moving_mean.Axpby(alpha, cur_mean, 1 - alpha);
	}
	
	output.MM(mul, moving_mean, Trans::N, Trans::N, -1.0, 1.0);

	DTensor<mode, Dtype> buf, cur_std;
	buf.CopyFrom(output);
	buf.Square();

	cur_std.MM(mul, buf, Trans::T, Trans::N, 1.0 / input.rows(), 0.0);
	//cur_std.InvSqrt();


}

template<typename mode, typename Dtype>
void BatchNorm<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 3, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad;
	if (isConst[0])
		return;

	auto& grad_0 = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad;

}

INSTANTIATE_CLASS(BatchNorm)

}