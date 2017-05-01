#include "nn/row_selection.h"
#include "tensor/mkl_helper.h"

namespace gnn
{

template<typename Dtype>
void RowSelectionFwd(DTensor<CPU, Dtype>& input, DTensor<CPU, Dtype>& output, DTensor<CPU, int>& row_idxes)
{
	output.Reshape({row_idxes.shape.Count(), input.cols()});
	Dtype* dst = output.data->ptr;
	for (size_t i = 0; i < row_idxes.shape.Count(); ++i)
	{
		Dtype* src = input.data->ptr + row_idxes.data->ptr[i] * input.cols();
		memcpy(dst, src, sizeof(Dtype) * input.cols());
		dst += input.cols();
	}
}

template<typename Dtype>
void RowSelectionBackwd(DTensor<CPU, Dtype>& prev_grad, DTensor<CPU, Dtype>& cur_grad, DTensor<CPU, int>& row_idxes)
{
	ASSERT( prev_grad.rows() == row_idxes.shape.Count(), "shape mismatch");

	const Dtype* src = cur_grad.data->ptr;

	for (size_t i = 0; i < row_idxes.shape.Count(); ++i)
	{
		Dtype* dst = prev_grad.data->ptr + row_idxes.data->ptr[i] * cur_grad.cols();

		MKL_Axpy(cur_grad.cols(), 1.0, src, dst);
		src += cur_grad.cols();
	}
}

template<typename mode, typename Dtype>
RowSelection<mode, Dtype>::RowSelection(std::string _name, PropErr _properr) 
		: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void RowSelection<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& idxes = dynamic_cast<DTensorVar<mode, int>*>(operands[1].get())->value;

	RowSelectionFwd(input, output, idxes);
}

template<typename mode, typename Dtype>
void RowSelection<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() >= 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad;

	auto& prev_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad;
	auto& idxes = dynamic_cast<DTensorVar<mode, int>*>(operands[1].get())->value;

	RowSelectionBackwd(prev_grad, cur_grad, idxes);
}

template class RowSelection<CPU, float>;
template class RowSelection<CPU, double>;
// template class RowSelection<GPU, float>;
// template class RowSelection<GPU, double>;
}