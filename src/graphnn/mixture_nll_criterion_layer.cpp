#include "mixture_nll_criterion_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
MixtureNLLCriterionLayer<mode, Dtype>::MixtureNLLCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr)
        : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr)
{    
    this->state = new DenseMat<mode, Dtype>();    
}        
        
template<MatMode mode, typename Dtype>
void MixtureNLLCriterionLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
    assert(operands.size() == 2);
    
    auto& q_k = operands[0]->state->DenseDerived();
    auto& log_p_xk = operands[1]->state->DenseDerived();
    auto& state = this->state->DenseDerived();
    
    state.EleWiseMul(q_k, log_p_xk);
    
    this->loss = -1.0 * state.Sum();                 
}

template<MatMode mode, typename Dtype>
void MixtureNLLCriterionLayer<mode, Dtype>::BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta)
{
    assert(operands.size() == 2);
    assert(cur_idx == 1);
        
    auto& q_k = operands[0]->state->DenseDerived();
    auto& prev_grad = operands[cur_idx]->grad->DenseDerived();
    
    auto& buf = this->state->DenseDerived();    
    buf.CopyFrom(q_k);
        
    buf.Scale(-this->lambda / buf.rows);
    
    if (beta == 0)
        prev_grad.CopyFrom(buf);
    else
        prev_grad.Axpby(1.0, buf, beta);                    
}

template class MixtureNLLCriterionLayer<CPU, float>;
template class MixtureNLLCriterionLayer<CPU, double>;
template class MixtureNLLCriterionLayer<GPU, float>;
template class MixtureNLLCriterionLayer<GPU, double>;
