#include "nn/row_selection.h"
#include "tensor/mkl_helper.h"
#include "tbb/tbb.h"

namespace gnn
{

template<typename Dtype>
void RowSelectionFwd(DTensor<CPU, Dtype>& input, DTensor<CPU, Dtype>& output, DTensor<CPU, int>& row_idxes)
{
	output.Reshape({row_idxes.shape.Count(), input.cols()});
	
	auto* src_ptr = row_idxes.data->ptr;

	size_t row_cnt = row_idxes.shape.Count();
	size_t dim = input.cols();

	tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
		Dtype* src = input.data->ptr + src_ptr[i] * dim;
		Dtype* dst = output.data->ptr + dim * i;
		memcpy(dst, src, sizeof(Dtype) * input.cols());
	});
}

template<typename Dtype>
void RowSelectionBackwd(RowSpTensor<CPU, Dtype>& prev_grad, DTensor<CPU, Dtype>& cur_grad, DTensor<CPU, int>& row_idxes)
{
	ASSERT( cur_grad.rows() == row_idxes.shape.Count(), "shape mismatch");
	ASSERT( !prev_grad.is_full, "suppose to be row sparse in RowSelectionBackwd");
	ASSERT( prev_grad.row_idxes.shape.Count() == 0, "only support one bp in RowSelectionBackwd");

	size_t row_cnt = row_idxes.shape.Count();
	size_t dim = cur_grad.cols();

	tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
		Dtype* dst = prev_grad.data->ptr + row_idxes.data->ptr[i] * dim;
		const Dtype* src = cur_grad.data->ptr + i * dim;

		MKL_Axpy(dim, 1.0, src, dst);
	});

	prev_grad.InsertRowIdxes(row_cnt, row_idxes.data->ptr);
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

	auto cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

	auto& prev_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad;
	auto& idxes = dynamic_cast<DTensorVar<mode, int>*>(operands[1].get())->value;

	RowSelectionBackwd(prev_grad, cur_grad, idxes);
}

template class RowSelection<CPU, float>;
template class RowSelection<CPU, double>;
// template class RowSelection<GPU, float>;
// template class RowSelection<GPU, double>;
}