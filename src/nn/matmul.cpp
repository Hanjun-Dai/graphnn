#include "nn/matmul.h"

namespace gnn
{

template<typename mode, typename Dtype>
MatMul<mode, Dtype>::MatMul(std::string _name, Trans _transA, Trans _transB, PropErr _properr) 
		: Factor(_name, _properr), transA(_transA), transB(_transB)
{

}

template<typename mode, typename Dtype>
void MatMul<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto* rhs = dynamic_cast< TensorVar<mode, Dtype>* >(operands[1].get());
	ASSERT(rhs->GetMatType() == MatType::dense, "only support [sparse/dense] x [dense]");

	auto* lhs = dynamic_cast< TensorVar<mode, Dtype>* >(operands[0].get());
	MAT_TYPE_SWITCH(lhs->GetMatType(), matType, {
		auto& left_mat = lhs->template Derived<matType>().value;
		auto& right_mat = rhs->template Derived<DENSE>().value;

		output.MM(left_mat, right_mat, transA, transB, 1.0, 0.0);
	});
}

template<typename mode, typename Dtype>
void MatMul<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto grad_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

	auto* rhs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get());
	auto* lhs = dynamic_cast< TensorVar<mode, Dtype>* >(operands[0].get());

	if (!isConst[1])
	{
		auto right_grad = rhs->grad.Full();
		if (transB == Trans::T)
		{
			ASSERT(lhs->GetMatType() == MatType::dense, "unsupported backprop in matmul");
			auto& left_mat = lhs->template Derived<DENSE>().value;
			right_grad.MM(grad_out, left_mat, Trans::T, transA, 1.0, 1.0);
		} else {
			MAT_TYPE_SWITCH(lhs->GetMatType(), matType, {
				auto& left_mat = lhs->template Derived<matType>().value;
			
				right_grad.MM(left_mat, grad_out, transA == Trans::N ? Trans::T : Trans::N, Trans::N, 1.0, 1.0);			
			});
		}
	}

	if (!isConst[0])
	{
		ASSERT(lhs->GetMatType() == MatType::dense, "differentiable lhs can't be sparse");
		auto left_grad = lhs->template Derived<DENSE>().grad.Full();
		auto& right_mat = rhs->value;
		
		if (transA == Trans::N)
			left_grad.MM(grad_out, right_mat, Trans::N, transB == Trans::N ? Trans::T : Trans::N, 1.0, 1.0);
		else
			left_grad.MM(right_mat, grad_out, transB, Trans::T, 1.0, 1.0);
	}
}

INSTANTIATE_CLASS(MatMul)

}