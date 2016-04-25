#include "gaussian_ll_layer.h"
#include "dense_matrix.h"

#define pi 3.14159265358979

template<MatMode mode, typename Dtype>
GaussianLLLayer<mode, Dtype>::GaussianLLLayer(std::string _name, PropErr _properr)
        : ILayer<mode, Dtype>(_name, _properr) 
{
    this->grad = new DenseMat<mode, Dtype>();
    this->state = new DenseMat<mode, Dtype>();    
}        
        
template<MatMode mode, typename Dtype>
void GaussianLLLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
    assert(operands.size() == 3);
    auto& mu = operands[0]->state->DenseDerived();
    auto& sigma = operands[1]->state->DenseDerived();
    auto& x = operands[2]->state->DenseDerived();
    
    // (x - mu)^2
    auto& state = this->state->DenseDerived(); 
    state.GeaM(1.0, Trans::N, mu, -1.0, Trans::N, x);
    state.Square();
    
    // (x - mu)^2 / 2 / sigma / sigma
    state.Scale(0.5);
    state.EleWiseDiv(sigma);
    state.EleWiseDiv(sigma);
    
    // (x - mu)^2 / 2 / sigma / sigma + 0.5 * log(2 pi)    
    state.Add(0.5 * log(2 * pi));
    
    // log(sigma) + (x - mu)^2 / 2 / sigma / sigma + 0.5 * log(2 pi)
    auto& grad = this->grad->DenseDerived();
    grad.Log(sigma);
    state.Axpy(1.0, grad);
    
    // log likelihood
    state.Scale(-1.0);
    
    if (this->properr == PropErr::T)
    {        
        diff.GeaM(1.0, Trans::N, mu, -1.0, Trans::N, x);
    }        
}

template<MatMode mode, typename Dtype>
void GaussianLLLayer<mode, Dtype>::BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta)
{
    assert(operands.size() == 3 && cur_idx <= 1);
        
    auto& cur_grad = this->grad->DenseDerived();                
    auto& prev_grad = operands[cur_idx]->grad->DenseDerived();
    auto& sigma = operands[1]->state->DenseDerived();
    
    if (cur_idx == 0)
    {
        buffer.CopyFrom(diff);
        buffer.EleWiseDiv(sigma);
        buffer.EleWiseDiv(sigma);
        buffer.Scale(-1.0);                                 
    } else 
    {
        buffer.CopyFrom(diff);
        buffer.EleWiseDiv(sigma); 
        buffer.Square();        
        buffer.Add(-1.0);
        
        buffer.EleWiseDiv(sigma);        
    }
    
    buffer.EleWiseMul(cur_grad);  
    if (beta == 0)
            prev_grad.CopyFrom(buffer);
        else
            prev_grad.Axpby(1.0, buffer, beta);
}

template class GaussianLLLayer<CPU, float>;
template class GaussianLLLayer<CPU, double>;
template class GaussianLLLayer<GPU, float>;
template class GaussianLLLayer<GPU, double>;
