#include "learner.h"

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