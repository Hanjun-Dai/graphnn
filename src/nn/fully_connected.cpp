#include "nn/fully_connected.h"

namespace gnn
{

template<typename mode, typename Dtype>
FullyConnected<mode, Dtype>::FullyConnected(std::string _name, PropErr _properr) 
		: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void FullyConnected<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& param = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;	

	auto* lhs = dynamic_cast< TensorVar<mode, Dtype>* >(operands[0].get());
	MAT_TYPE_SWITCH(lhs->GetMatType(), matType, {
		auto& input = lhs->template Derived<matType>().value;
		auto weight = param.GetRowRef(0, input.cols());

		output.MM(input, weight, Trans::N, Trans::N, 1.0, 0.0);
		if (param.rows() > input.cols())
		{
			ASSERT(param.rows() == input.cols() + 1, "size mismatch");
			auto bias = param.GetRowRef(input.cols(), 1);

			auto mul = DTensor<mode, Dtype>({input.rows(), (size_t)1});
			mul.Fill(1.0);
			output.MM(mul, bias, Trans::N, Trans::N, 1.0, 1.0);
		}
	});
}

template<typename mode, typename Dtype>
void FullyConnected<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto grad_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

	auto* rhs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get());
	auto param_grad = rhs->grad.Full();

	auto* lhs = dynamic_cast< TensorVar<mode, Dtype>* >(operands[0].get());

	if (!isConst[1])
	{
		MAT_TYPE_SWITCH(lhs->GetMatType(), matType, {
			auto& input = lhs->template Derived<matType>().value;
			auto weight_grad = param_grad.GetRowRef(0, input.cols());		

			weight_grad.MM(input, grad_out, Trans::T, Trans::N, 1.0, 1.0);
			if (param_grad.rows() > input.cols())
			{
				auto bias_grad = param_grad.GetRowRef(input.cols(), 1);

				auto mul = DTensor<mode, Dtype>({input.rows(), (size_t)1});
				mul.Fill(1.0);

				bias_grad.MM(mul, grad_out, Trans::T, Trans::N, 1.0, 1.0);
			}
		});	
	}

	if (!isConst[0])
	{
		ASSERT(lhs->GetMatType() == MatType::dense, "differentiable lhs can't be sparse");
		auto input_grad = lhs->template Derived<DENSE>().grad.Full();
		auto weight = rhs->value.GetRowRef(0, input_grad.cols());

		input_grad.MM(grad_out, weight, Trans::N, Trans::T, 1.0, 1.0);
	}
}

INSTANTIATE_CLASS(FullyConnected)

}