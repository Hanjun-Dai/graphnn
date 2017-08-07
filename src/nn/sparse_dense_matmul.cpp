#include "nn/sparse_dense_matmul.h"

namespace gnn
{

template<typename mode, typename Dtype>
SparseDenseMatMul<mode, Dtype>::SparseDenseMatMul(std::string _name, Trans _transA, Trans _transB, PropErr _properr) 
		: Factor(_name, _properr), transA(_transA), transB(_transB)
{

}

template<typename mode, typename Dtype>
void SparseDenseMatMul<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& rhs = dynamic_cast< DTensorVar<mode, Dtype>* >(operands[1].get())->value;

	auto& lhs = dynamic_cast< SpTensorVar<mode, Dtype>* >(operands[0].get())->value;

	output.MM(lhs, rhs, transA, transB, 1.0, 0.0);
}

template<typename mode, typename Dtype>
void SparseDenseMatMul<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto grad_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

	auto& right_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->grad;

	auto& left_mat = dynamic_cast< SpTensorVar<mode, Dtype>* >(operands[0].get())->value;

	if (!isConst[1])
	{
		ASSERT(transB == Trans::N, "unsupported backprop in matmul");

		right_grad.SparseMM(left_mat, grad_out, transA == Trans::N ? Trans::T : Trans::N, Trans::N, 1.0, 1.0);			
	}
}

INSTANTIATE_CLASS(SparseDenseMatMul)

}