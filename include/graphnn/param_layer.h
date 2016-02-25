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
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        
    }
    
    IParam<mode, Dtype>* param;        
};

#endif