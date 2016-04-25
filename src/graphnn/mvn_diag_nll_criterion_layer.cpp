#include "mvn_diag_nll_criterion_layer.h"
#include "dense_matrix.h"

#define pi 3.14159265358979

template<MatMode mode, typename Dtype>
MVNDianNLLCriterionLayer<mode, Dtype>::MVNDianNLLCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr)
        : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr)
{
    this->grad = new DenseMat<mode, Dtype>();
    this->state = new DenseMat<mode, Dtype>();    
}        
        
template<MatMode mode, typename Dtype>
void MVNDianNLLCriterionLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
    assert(operands.size() == 3);
    auto& mu = operands[0]->state->DenseDerived();
    auto& sigma = operands[1]->state->DenseDerived();
    auto& x = operands[2]->state->DenseDerived();
    
    // (x - mu)^2
    auto& buf = this->state->DenseDerived(); 
    buf.GeaM(1.0, Trans::N, mu, -1.0, Trans::N, x);
    buf.Square();
    
    // (x - mu)^2 / 2 / sigma / sigma
    buf.Scale(0.5);
    buf.EleWiseDiv(sigma);
    buf.EleWiseDiv(sigma);
    
    // (x - mu)^2 / 2 / sigma / sigma + 0.5 * log(2 pi)    
    buf.Add(0.5 * log(2 * pi));
    
    // log(sigma) + (x - mu)^2 / 2 / sigma / sigma + 0.5 * log(2 pi)
    auto& grad = this->grad->DenseDerived();
    grad.Log(sigma);
    buf.Axpy(1.0, grad);
        
    this->loss = buf.Sum();
    
    if (this->properr == PropErr::T)
        grad.GeaM(1.0, Trans::N, mu, -1.0, Trans::N, x);
}

template<MatMode mode, typename Dtype>
void MVNDianNLLCriterionLayer<mode, Dtype>::BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta)
{
    assert(operands.size() == 3 && cur_idx <= 1);
            
    auto& diff = this->grad->DenseDerived();
    auto& buf = this->state->DenseDerived();
    auto& prev_grad = operands[cur_idx]->grad->DenseDerived();
    auto& sigma = operands[1]->state->DenseDerived();    
    
    // d/d_mu
    if (cur_idx == 0)
    {
        buf.CopyFrom(diff);
        buf.EleWiseDiv(sigma);
        buf.EleWiseDiv(sigma);
        buf.Scale(this->lambda / buf.rows);
        
        if (beta == 0)
            prev_grad.CopyFrom(buf);
        else
            prev_grad.Axpby(1.0, buf, beta);       
    } else // d/d_sigma
    {
        // 1/sigma
        buf.CopyFrom(sigma);
        buf.Inv();
        
        if (beta == 0)
        {
            prev_grad.CopyFrom(buf);
            prev_grad.Scale(this->lambda / buf.rows);
        }
        else
            prev_grad.Axpby(1.0, buf, beta);
        
        // (x - mu) ^2 / sigma^3    
        buf.Power(3);
        buf.EleWiseMul(diff);
        buf.EleWiseMul(diff);
        
        prev_grad.Axpy(-this->lambda / buf.rows, buf);
    }
}

template class MVNDianNLLCriterionLayer<CPU, float>;
template class MVNDianNLLCriterionLayer<CPU, double>;
template class MVNDianNLLCriterionLayer<GPU, float>;
template class MVNDianNLLCriterionLayer<GPU, double>;
