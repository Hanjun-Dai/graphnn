#include "nngraph.h"
#include <cstring>
#include "param_layer.h"
#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
void NNGraph<mode, Dtype>::ForwardData(std::map<std::string, IMatrix<mode, Dtype>* > input, Phase phase)
{
    if (phase == TRAIN)
    {
        for (auto item : layer_dict)
            item.second->ClearGrad();
    }
    // feed data
    for (auto it = input.begin(); it != input.end(); ++it)
    {
        layer_dict[it->first]->state = it->second;
    }
    
    // feed-forward
    for (size_t i = 0; i < ordered_layers.size(); ++i)
    {     
        auto* cur_layer = layer_dict[ordered_layers[i].first];
        auto& operands = ordered_layers[i].second;
        
        cur_layer->UpdateOutput(operands, phase);        
    }    
}

template<MatMode mode, typename Dtype>
std::map<std::string, Dtype> NNGraph<mode, Dtype>::ForwardLabel(std::map<std::string, IMatrix<mode, Dtype>* > ground_truth)
{
    std::map<std::string, Dtype> loss;
    has_grad.resize(ordered_layers.size());
    for (bool&& g : has_grad)
        g = false;
    
    for (auto it = ground_truth.begin(); it != ground_truth.end(); ++it)
    {
        auto* criterion_layer = dynamic_cast<ICriterionLayer<mode, Dtype>*>(layer_dict[it->first]);
        assert(criterion_layer);
        loss[it->first] = criterion_layer->GetLoss(it->second);         
        has_grad[name_idx_map[it->first]] = (criterion_layer->properr == PropErr::T);
    }
    return loss;
}

template<MatMode mode, typename Dtype>
void NNGraph<mode, Dtype>::BackPropagation()
{    	
    for (auto it = ordered_layers.rbegin(); it != ordered_layers.rend(); ++it)
    {
        auto* cur_layer = layer_dict[it->first];
        auto& operands = it->second;
        
        if (cur_layer->properr != PropErr::T)
			continue;
        if (!has_grad[ name_idx_map[cur_layer->name] ])
            continue;
        
        for (size_t i = 0; i < operands.size(); ++i)
        {
            auto* prev_layer = operands[i];
            if (prev_layer->properr == PropErr::T)
            {
                cur_layer->BackPropErr(operands, i);
                has_grad[ name_idx_map[prev_layer->name] ] = true;
            }
            if (cur_layer->HasParam())
                dynamic_cast<ParamLayer<mode, Dtype>*>(cur_layer)->AccDeriv(operands, i);
        }                          
    }    
}

template class NNGraph<CPU, float>;
template class NNGraph<CPU, double>;
template class NNGraph<GPU, float>;
template class NNGraph<GPU, double>;