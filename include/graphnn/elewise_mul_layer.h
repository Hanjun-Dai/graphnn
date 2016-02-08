#ifndef ELEWISE_MUL_LAYER_H
#define ELEWISE_MUL_LAYER_H

#include "ilayer.h"
#include <map>

template<MatMode mode, typename Dtype>
class ElewiseMulLayer : public ILayer<mode, Dtype>
{
public:
    ElewiseMulLayer(std::string _name, GraphAtt _at, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _at, _properr)
    {
        this->graph_output = new GraphData<mode, Dtype>(DENSE);
        this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
    }
      
    virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override
    {
        ILayer<mode, Dtype>::UpdateOutput(prev_layer, sv, phase);
        
        auto& cur_output = GetImatState(this->graph_output, this->at)->DenseDerived();      
        auto& prev_states = GetImatState(prev_layer->graph_output, this->at)->DenseDerived();        
        
        if (sv == SvType::WRITE2)
            cur_output.CopyFrom(prev_states);
        else 
            cur_output.EleWiseMul(prev_states);  
    }
    
    virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
    {
        // not implemented for multiple (>2) inputs
        assert(false);    
    }
};

template<MatMode mode, typename Dtype>
class PairMulLayer : public ElewiseMulLayer<mode, Dtype>
{
public:
    PairMulLayer(std::string _name, GraphAtt _at, PropErr _properr = PropErr::T)
        : ElewiseMulLayer<mode, Dtype>(_name, _at, _properr)
    {
        prev_layer_names.clear();
    }
    
    virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override
    {
        ElewiseMulLayer<mode, Dtype>::UpdateOutput(prev_layer, sv, phase);
        prev_layer_names[prev_layer->name] = prev_layer;
    }
    
    virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
    {
        assert(prev_layer_names.size() == 2);
        assert(prev_layer_names.count(prev_layer->name));
        auto& cur_grad = GetImatState(this->graph_gradoutput, this->at)->DenseDerived();
        auto& prev_grad = GetImatState(prev_layer->graph_gradoutput, this->at)->DenseDerived(); 
        
        for (auto it = prev_layer_names.begin(); it != prev_layer_names.end(); ++it)
        {
            if (it->first == prev_layer->name)
                continue;
            
            auto& another_state = GetImatState(it->second->graph_output, this->at)->DenseDerived();
            if (sv == SvType::WRITE2)
            {                
                prev_grad.EleWiseMul(cur_grad, another_state);
            }
            else 
            {
                buf.EleWiseMul(cur_grad, another_state);
                prev_grad.Axpy(1.0, buf);
            }
        }
    }
    
    std::map<std::string, ILayer<mode, Dtype>*> prev_layer_names;
    DenseMat<mode, Dtype> buf;
};

#endif