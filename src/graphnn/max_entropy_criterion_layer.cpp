#include "max_entropy_criterion_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
MaxEntropyCriterionLayer<mode, Dtype>::MaxEntropyCriterionLayer(std::string _name, 
                                                                  Dtype _lambda, 
                                                                  PropErr _properr)
				 : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr)
{
    this->state = new DenseMat<mode, Dtype>();
    this->grad = new DenseMat<mode, Dtype>();
}

template<MatMode mode, typename Dtype>
void MaxEntropyCriterionLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
    assert(operands.size() == 1);       
    
    auto& p = operands[0]->state->DenseDerived();
    
    auto& grad = this->grad->DenseDerived();
    grad.Log(p);
    
    auto& state = this->state->DenseDerived();
    state.EleWiseMul(p, grad);    
    
    // negative entropy, we assume to maximize the entropy here
    this->loss = state.Sum();                                  
}

template<MatMode mode, typename Dtype>
void MaxEntropyCriterionLayer<mode, Dtype>::BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta)
{
    assert(operands.size() == 1 && cur_idx == 0);
    
    auto& grad = this->grad->DenseDerived();
    auto& prev_grad = operands[cur_idx]->grad->DenseDerived();
    
    if (beta == 0)
    {
        prev_grad.CopyFrom(grad);
        prev_grad.Scale(this->lambda / grad.rows);
    }
    else
        prev_grad.Axpby(this->lambda / grad.rows, grad, beta);
        
    prev_grad.Add(this->lambda / grad.rows);                          
}

template class MaxEntropyCriterionLayer<CPU, float>;
template class MaxEntropyCriterionLayer<CPU, double>;
template class MaxEntropyCriterionLayer<GPU, float>;
template class MaxEntropyCriterionLayer<GPU, double>;
