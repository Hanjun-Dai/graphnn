#include "nn/concat_cols.h"

namespace gnn
{
template<typename mode, typename Dtype>
ConcatCols<mode, Dtype>::ConcatCols(std::string _name, PropErr _properr) 
		: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void ConcatCols<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() >= 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

    std::vector< DTensor<mode, Dtype>* > prev_states;
    prev_states.resize(operands.size());
    
    for (size_t i = 0; i < operands.size(); ++i)
        prev_states[i] = &(dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->value);
    output.ConcatCols(prev_states); 
}

template<typename mode, typename Dtype>
void ConcatCols<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() >= 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();
	size_t col_start = 0;

	DTensor<mode, Dtype> buf;		
	for (size_t i = 0; i < operands.size(); ++i)
	{		
		if (!isConst[i])
		{
			auto grad_i = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->grad.Full();
			buf.CopyColsFrom(cur_grad, col_start, grad_i.cols());
			grad_i.Axpy(1.0, buf);
		}
		auto& state_i = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->value;
		col_start += state_i.cols();
	}
}

INSTANTIATE_CLASS(ConcatCols)

}