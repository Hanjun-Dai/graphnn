#include "nn/multi_matmul.h"

namespace gnn
{

template<typename mode, typename Dtype>
MultiMatMul<mode, Dtype>::MultiMatMul(std::string _name, PropErr _properr) 
		: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void MultiMatMul<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() % 2 == 0, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	for (size_t i = 0; i < operands.size(); i += 2)
	{
		auto* rhs = dynamic_cast< TensorVar<mode, Dtype>* >(operands[i + 1].get());
		ASSERT(rhs->GetMatType() == MatType::dense, "only support [sparse/dense] x [dense]");

		auto* lhs = dynamic_cast< TensorVar<mode, Dtype>* >(operands[i].get());
		MAT_TYPE_SWITCH(lhs->GetMatType(), matType, {
			auto& left_mat = lhs->template Derived<matType>().value;
			auto& right_mat = rhs->template Derived<DENSE>().value;

			output.MM(left_mat, right_mat, Trans::N, Trans::N, 1.0, i > 0);
		});
	}
}

template<typename mode, typename Dtype>
void MultiMatMul<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() % 2 == 0, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto grad_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

	for (size_t i = 0; i < operands.size(); i += 2)
	{
		auto* rhs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i + 1].get());
		auto right_grad = rhs->grad.Full();
		auto* lhs = dynamic_cast< TensorVar<mode, Dtype>* >(operands[i].get());

		if (!isConst[i + 1])
		{
			MAT_TYPE_SWITCH(lhs->GetMatType(), matType, {
				auto& left_mat = lhs->template Derived<matType>().value;
			
				right_grad.MM(left_mat, grad_out, Trans::T, Trans::N, 1.0, 1.0);			
			});
		}
		if (!isConst[0])
		{
			ASSERT(lhs->GetMatType() == MatType::dense, "differentiable lhs can't be sparse");
			auto left_grad = lhs->template Derived<DENSE>().grad.Full();
			auto& right_mat = rhs->value;
			
			left_grad.MM(grad_out, right_mat, Trans::N, Trans::T, 1.0, 1.0);
		}
	}
}

INSTANTIATE_CLASS(MultiMatMul)

}