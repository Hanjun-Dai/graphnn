#include "global_sum_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
GlobalSumLayer<mode, Dtype>::GlobalSumLayer(std::string _name, PropErr _properr)
                : ILayer<mode, Dtype>(_name, _properr)
{
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();    
}   

template<MatMode mode, typename Dtype>
void GlobalSumLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
        assert(operands.size() == 1);
        
        auto& cur_output = this->state->DenseDerived();
        auto& prev_state = operands[0]->state->DenseDerived();
        
        this->buf.Resize(prev_state.cols, 1);
        this->buf.Fill(1.0);
        
        cur_output.GeMM(prev_state, buf, Trans::N, Trans::N, 1.0, 0);
}

template<MatMode mode, typename Dtype>
void GlobalSumLayer<mode, Dtype>::BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta)
{
        assert(operands.size() == 1 && cur_idx == 0);
        
        auto& cur_grad = this->grad->DenseDerived();
        auto& prev_grad = operands[0]->grad->DenseDerived();
        
        prev_grad.GeMM(cur_grad, this->buf, Trans::N, Trans::T, 1.0, beta);        
}

template class GlobalSumLayer<CPU, double>;
template class GlobalSumLayer<CPU, float>;
template class GlobalSumLayer<GPU, double>;
template class GlobalSumLayer<GPU, float>;
