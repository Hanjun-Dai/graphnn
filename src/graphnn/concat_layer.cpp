#include "concat_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
ConcatLayer<mode, Dtype>::ConcatLayer(std::string _name, PropErr _properr)
                            : ILayer<mode, Dtype>(_name, _properr)
{
    this->state = new DenseMat<mode, Dtype>();
    this->grad = new DenseMat<mode, Dtype>();
}

template<MatMode mode, typename Dtype>
void ConcatLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
    std::vector< DenseMat<mode, Dtype>* > prev_states;
    prev_states.resize(operands.size());
    
    for (size_t i = 0; i < operands.size(); ++i)
        prev_states[i] = &(operands[i]->state->DenseDerived());
        
    auto& state = this->state->DenseDerived();
    state.ConcatCols(prev_states);        
}

template<MatMode mode, typename Dtype>
void ConcatLayer<mode, Dtype>::BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta)
{
    assert(cur_idx < operands.size());
    
    auto& prev_grad = operands[cur_idx]->grad->DenseDerived();
    auto& grad = this->grad->DenseDerived();
    
    size_t col_start = 0;
    for (size_t i = 0; i < cur_idx; ++i)
        col_start += operands[i]->state->cols;            
        
    buf.GetColsFrom(grad, col_start, operands[cur_idx]->state->cols);
        
    if (beta == 0)
        prev_grad.CopyFrom(buf);
    else
        prev_grad.Axpby(1.0, buf, beta);
}

template class ConcatLayer<CPU, double>;
template class ConcatLayer<CPU, float>;
template class ConcatLayer<GPU, double>;
template class ConcatLayer<GPU, float>;
