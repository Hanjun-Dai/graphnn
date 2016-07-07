#include "learner.h"
#include <cmath>

template<MatMode mode, typename Dtype>
Dtype ILearner<mode, Dtype>::ClipGradients()
{
    if (this->clipping_enabled)
    {
        auto& param_list = this->model->GetDiffParams();
                
        Dtype norm = 0.0;
        for (auto& param_pair : param_list)
        {
            auto* param = param_pair.second;
            Dtype norm2 = param->grad.Norm2();
            norm += norm2 * norm2;
        }
        norm = sqrt(norm);
        if (norm > this->clip_threshold)
            return this->clip_threshold / norm;
    }
    return 1.0;
}

template class ILearner<CPU, float>;
template class ILearner<GPU, float>;
template class ILearner<CPU, double>;
template class ILearner<GPU, double>;

template<MatMode mode, typename Dtype>
void SGDLearner<mode, Dtype>::Update()
{
    auto& param_list = this->model->GetDiffParams();
            
    for (auto& param_pair : param_list)
    {
        auto* param = param_pair.second;
        param->value.Axpby(-this->cur_lr, param->grad, 1 - this->cur_lr * this->l2_penalty);
        param->grad.Zeros();
    }
}

template class SGDLearner<CPU, float>;
template class SGDLearner<GPU, float>;
template class SGDLearner<CPU, double>;
template class SGDLearner<GPU, double>;

template<MatMode mode, typename Dtype>
void MomentumSGDLearner<mode, Dtype>::Update()
{
    auto& param_list = this->model->GetDiffParams();
            
    for (auto& param_pair : param_list)
    {        
        auto& name = param_pair.first;
        auto* param = param_pair.second;        
        if (momentum > 0)
        {
            if (acc_grad_dict.count(name) == 0)
            {
                acc_grad_dict[name] = std::make_shared< DenseMat<mode, Dtype> >(param->grad.rows, param->grad.cols);
                acc_grad_dict[name]->Zeros();
            }
            param->grad.Axpy(this->l2_penalty, param->value);
            acc_grad_dict[name]->Axpby(this->cur_lr, param->grad, momentum);
            param->value.Axpy(-1.0, *acc_grad_dict[name]);
        } else // do normal sgd
            param->value.Axpby(-this->cur_lr, param->grad, 1 - this->cur_lr * this->l2_penalty);
        param->grad.Zeros();
    }    
}

template class MomentumSGDLearner<CPU, float>;
template class MomentumSGDLearner<GPU, float>;
template class MomentumSGDLearner<CPU, double>;
template class MomentumSGDLearner<GPU, double>;

template<MatMode mode, typename Dtype>
void ExplicitBatchLearner<mode, Dtype>::Update()
{
    auto& param_list = this->model->GetDiffParams();
            
    for (auto& param_pair : param_list)
    {
        auto name = param_pair.first;
        auto* param = param_pair.second;
        
        assert(acc_grad_dict.count(name));
        
        param->value.Axpby(-this->cur_lr, *(acc_grad_dict[name]), 1 - this->cur_lr * this->l2_penalty);
        param->grad.Zeros();
        acc_grad_dict[name]->Zeros();
    }    
}

template<MatMode mode, typename Dtype>
void ExplicitBatchLearner<mode, Dtype>::AccumulateGrad()
{
    auto& param_list = this->model->GetDiffParams();
            
    for (auto& param_pair : param_list)
    {
        auto name = param_pair.first;
        auto* param = param_pair.second;
        if (acc_grad_dict.count(name) == 0)
        {
            acc_grad_dict[name] = std::make_shared< DenseMat<mode, Dtype> >(param->grad.rows, param->grad.cols);
            acc_grad_dict[name]->Zeros();
        }
        
        acc_grad_dict[name]->Axpy(1.0, param->grad);              
        param->grad.Zeros();
    }
}

template class ExplicitBatchLearner<CPU, float>;
template class ExplicitBatchLearner<GPU, float>;
template class ExplicitBatchLearner<CPU, double>;
template class ExplicitBatchLearner<GPU, double>;

template<MatMode mode, typename Dtype>
void AdamLearner<mode, Dtype>::Update()
{
    auto& param_list = this->model->GetDiffParams();
            
    Dtype gscale = this->ClipGradients();
    this->cur_iter++;

    for (auto& param_pair : param_list)
    {
        auto name = param_pair.first;
        auto* param = param_pair.second;
        if (first_moments.count(name) == 0 && second_moments.count(name) == 0)
        {
            first_moments[name] = std::make_shared< DenseMat<mode, Dtype> >(param->grad.rows, param->grad.cols);
            first_moments[name]->Zeros();
            second_moments[name] = std::make_shared< DenseMat<mode, Dtype> >(param->grad.rows, param->grad.cols);
            second_moments[name]->Zeros();
        }           
        assert(first_moments.count(name) && second_moments.count(name));
        auto& m_t = *(first_moments[name]); 
        auto& v_t = *(second_moments[name]);
        // clipping and weight decay
        param->grad.Axpby(this->l2_penalty, param->value, gscale);
        // m_t = beta_1 * m_{t-1} + (1 - beta_1) * gt
        m_t.Axpby(1 - beta_1, param->grad, beta_1);
        // v_t = beta_2 * v_{t-1} + (1 - beta_2) * gt^2
        param->grad.Square();
        v_t.Axpby(1 - beta_2, param->grad, beta_2);

        // 1 / (1 - beta^t)
        Dtype s1 = 1.0 / (1 - pow(beta_1, this->cur_iter));
        Dtype s2 = 1.0 / (1 - pow(beta_2, this->cur_iter)); 

        // v_hat = 1 ./ (sqrt(v_t / (1 - beta_2^t)) + eps)
        v_hat.CopyFrom(v_t);
        v_hat.Scale(s2);
        v_hat.Sqrt();
        v_hat.Add(eps);
        v_hat.Inv();

        v_hat.EleWiseMul(m_t);
        param->value.Axpby(-this->cur_lr * s1, v_hat, 1.0);

        param->grad.Zeros();
    }
}

template class AdamLearner<CPU, float>;
template class AdamLearner<GPU, float>;
template class AdamLearner<CPU, double>;
template class AdamLearner<GPU, double>;

