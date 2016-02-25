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
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        assert(operands.size() == 1);
        auto* prev_layer = operands[0];        
        auto& cur_output = this->state->DenseDerived();
        
        if (param->OutSize()) // if we can know the outputsize ahead
            cur_output.Resize(prev_layer->state->rows, param->OutSize());
        
        param->UpdateOutput(prev_layer->state, &cur_output, phase);        
    }
    
    IParam<mode, Dtype>* param;        
};

#endif