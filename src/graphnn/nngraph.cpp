#include "nngraph.h"

template<MatMode mode, typename Dtype>
void NNGraph<mode, Dtype>::ForwardData(std::map<std::string, IMatrix<mode, Dtype>* > input, Phase phase)
{
    
}


template<MatMode mode, typename Dtype>
std::map<std::string, Dtype> NNGraph<mode, Dtype>::ForwardLabel(std::map<std::string, IMatrix<mode, Dtype>* > ground_truth)
{
    
}

template<MatMode mode, typename Dtype>
void NNGraph<mode, Dtype>::BackPropagation()
{
    
}

template class NNGraph<CPU, float>;
template class NNGraph<CPU, double>;
template class NNGraph<GPU, float>;
template class NNGraph<GPU, double>;