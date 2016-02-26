#ifndef PARAM_LAYER_H
#define PARAM_LAYER_H

#include "i_layer.h"
#include "i_param.h"

template<MatMode mode, typename Dtype>
class ParamLayer : public ILayer<mode, Dtype>
{
public:
    ParamLayer(std::string _name, IParam<mode, Dtype>* _param, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _properr) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
        this->param = _param;
    }

    static std::string str_type()
    {
        return "Param"; 
    }
    
    virtual bool HasParam() override
    {
        return true;
    }
       
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        assert(operands.size() == 1);
        auto* prev_layer = operands[0];        
        auto& cur_output = this->state->DenseDerived();
        
        if (param->OutSize()) // if we can know the outputsize ahead
            cur_output.Resize(prev_layer->state->rows, param->OutSize());
        
        param->UpdateOutput(prev_layer->state, &cur_output, phase);        
    }
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx) override
    {
        assert(operands.size() == 1 && cur_idx == 0);
        
		auto& cur_grad = this->grad->DenseDerived();
        auto* prev_layer = operands[0];
        if (prev_grad.rows != cur_grad.rows && param->InSize()) // if we can know the inputsize ahead
            prev_grad.Resize(cur_grad.rows, param->InSize());		
                		
        auto& prev_grad = prev_layer->grad->DenseDerived(); 
		
		param->UpdateGradInput(&prev_grad, &cur_grad);        
    }
    
    void AccDeriv(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx)
    {
        
    }
    
    IParam<mode, Dtype>* param;        
};

#endif