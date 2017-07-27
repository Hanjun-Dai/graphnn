#include "nn/moving_norm.h"

namespace gnn
{

template<typename mode, typename Dtype>
MovingNorm<mode, Dtype>::MovingNorm(std::string _name, Dtype _alpha, PropErr _properr) 
		: Factor(_name, _properr), alpha(_alpha)
{
	eps = 1e-5;
	has_first = false;
}

template<typename mode, typename Dtype>
void MovingNorm<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 3, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& moving_mean = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;
	auto& moving_inv_std = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[2].get())->value;

	output.CopyFrom(input);
	auto mul = DTensor<mode, Dtype>({input.rows(), (size_t)1});
	mul.Fill(1.0);
        auto cur_alpha = alpha;
        if (!has_first)
        {
                has_first = true;
                cur_alpha = 1.0;
        }
	if (phase == Phase::TRAIN)
	{
		DTensor<mode, Dtype> cur_mean;
		cur_mean.MM(mul, input, Trans::T, Trans::N, 1.0 / input.rows(), 0.0);

		moving_mean.Axpby(cur_alpha, cur_mean, 1 - cur_alpha);
	}
	
	output.MM(mul, moving_mean, Trans::N, Trans::N, -1.0, 1.0);

	if (phase == Phase::TRAIN)
	{
		DTensor<mode, Dtype> buf, cur_inv_std;
		buf.CopyFrom(output);
		buf.Square();

		cur_inv_std.MM(mul, buf, Trans::T, Trans::N, 1.0 / input.rows(), 0.0);
		cur_inv_std.Add(eps);
		cur_inv_std.InvSqrt();

		moving_inv_std.Axpby(cur_alpha, cur_inv_std, 1 - cur_alpha);	
	}
	
	normed_inv_std.CopyFrom(moving_inv_std);
	normed_inv_std.Truncate(0, 1.0);

	output.ElewiseMul(normed_inv_std);
}

template<typename mode, typename Dtype>
void MovingNorm<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 3, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();
	if (isConst[0])
		return;

	auto grad_0 = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();

	DTensor<mode, Dtype> buf;
	buf.CopyFrom(cur_grad);
	buf.ElewiseMul(normed_inv_std);

	grad_0.Axpy(1.0, buf);
}

INSTANTIATE_CLASS(MovingNorm)

}
