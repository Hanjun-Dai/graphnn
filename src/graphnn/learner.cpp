#include "learner.h"

template<MatMode mode, typename Dtype>
void SGDLearner<mode, Dtype>::Update()
{
    auto& param_list = this->model->GetDiffParams();
            
    for (auto* param : param_list)
    {
        param->value.Axpby(-this->cur_lr, param->grad, 1 - this->cur_lr * this->l2_penalty);
    }    
}

template class SGDLearner<CPU, float>;
template class SGDLearner<GPU, float>;
template class SGDLearner<CPU, double>;
template class SGDLearner<GPU, double>;