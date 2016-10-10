#include "nngraph.h"
#include <cstring>
#include "param_layer.h"
#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
void NNGraph<mode, Dtype>::FeedForward(std::map<std::string, IMatrix<mode, Dtype>* > input, Phase phase)
{
    hash.resize(ordered_layers.size());
    for (size_t i = 0; i < hash.size(); ++i)
        hash[i] = false;
        
    // feed data
    for (auto it = input.begin(); it != input.end(); ++it)
    {
        layer_dict[it->first]->state = it->second;
        assert(name_idx_map.count(it->first));
        hash[name_idx_map[it->first]] = true;
    }
    
    // feed-forward
    for (size_t i = 0; i < ordered_layers.size(); ++i)
    {     
        assert(layer_dict.count(ordered_layers[i].first));
        auto* cur_layer = layer_dict[ordered_layers[i].first];
        auto& operands = ordered_layers[i].second;
        assert(name_idx_map.count(cur_layer->name));
        if (operands.size() == 0 && ! hash[name_idx_map[cur_layer->name]])
            continue;
        bool ready = true;
        for (auto* layer : operands)
        {
            if (static_layer_dict.count(layer->name))
                continue;
            assert(name_idx_map.count(layer->name));
            auto idx = name_idx_map[layer->name];
            ready &= hash[idx];
        }
        hash[name_idx_map[cur_layer->name]] = ready;
        if (ready)            
            cur_layer->UpdateOutput(operands, phase);
        else if (phase != TEST)
            throw std::runtime_error("wrong computation flow");         
    }    
}

template<MatMode mode, typename Dtype>
std::map<std::string, Dtype> NNGraph<mode, Dtype>::GetLoss()
{
    std::map<std::string, Dtype> loss;
            
    for (auto it = ordered_layers.rbegin(); it != ordered_layers.rend(); ++it)
    {
        auto* cur_layer = layer_dict[it->first];
        
        if (!cur_layer->IsSupervised())
            continue;
        
        auto* criterion_layer = dynamic_cast<ICriterionLayer<mode, Dtype>*>(cur_layer);
        assert(criterion_layer);
        loss[it->first] = criterion_layer->GetLoss(); 
    }
    return loss;
}

template<MatMode mode, typename Dtype>
void NNGraph<mode, Dtype>::BackPropagation()
{    	
    hash.resize(ordered_layers.size());
    for (size_t i = 0; i < hash.size(); ++i)
        hash[i] = false;
                        
    for (auto it = ordered_layers.rbegin(); it != ordered_layers.rend(); ++it)
    {
        auto* cur_layer = layer_dict[it->first];
        auto& operands = it->second;
        
        if (cur_layer->properr != PropErr::T)
			continue;
        if (!hash[ name_idx_map[cur_layer->name] ])
        {
            if (cur_layer->IsSupervised())
                hash[ name_idx_map[cur_layer->name] ] = true;
            else continue;
        }
        
        for (size_t i = 0; i < operands.size(); ++i)
        {            
            auto* prev_layer = operands[i];
            assert(name_idx_map.count(prev_layer->name));
            auto prev_id = name_idx_map[prev_layer->name];
            if (prev_layer->properr == PropErr::T)
            {
                Dtype beta = 1.0;
                // if we haven't backprop the error to this layer
                if (! hash[ prev_id ])
                {
                    beta = 0.0;
                    hash[prev_id] = true;
                    prev_layer->grad->DenseDerived().Zeros(prev_layer->state->rows, prev_layer->state->cols);
                }
                cur_layer->BackPropErr(operands, i, beta);
            }
            if (cur_layer->HasParam())
                dynamic_cast<IParametric<mode, Dtype>*>(cur_layer)->AccDeriv(operands, i);
        }                          
    }        
}

template class NNGraph<CPU, float>;
template class NNGraph<CPU, double>;
template class NNGraph<GPU, float>;
template class NNGraph<GPU, double>;