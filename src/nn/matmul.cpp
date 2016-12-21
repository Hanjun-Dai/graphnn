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
						 			std::vector< std::shared_ptr<Variable> >& outputs)
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
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& grad_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad;

	auto* rhs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get());

	auto* lhs = dynamic_cast< TensorVar<mode, Dtype>* >(operands[0].get());

	MAT_TYPE_SWITCH(lhs->GetMatType(), matType, {
		auto& left_mat = lhs->template Derived<matType>().value;
		auto& right_grad = rhs->grad;

		if (transB == Trans::N)
			right_grad.MM(left_mat, grad_out, transA == Trans::N ? Trans::T : Trans::N, Trans::N, 1.0, 1.0);
		else
			right_grad.MM(grad_out, left_mat, Trans::T, transA, 1.0, 1.0);
	});

}

template class MatMul<CPU, float>;
template class MatMul<CPU, double>;

}