#include "learner.h"

template<MatMode mode, typename Dtype>
void SGDLearner<mode, Dtype>::Update()
{
    auto& param_list = this->model->GetDiffParams();
            
    for (auto* param : param_list)
    {
        if (this->l2_penalty == 0)
        {
               
        } else {
                   
        }
    }    
}

template class SGDLearner<CPU, float>;
template class SGDLearner<GPU, float>;
template class SGDLearner<CPU, double>;
template class SGDLearner<GPU, double>;