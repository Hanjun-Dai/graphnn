#ifndef PARAM_LAYER_H
#define PARAM_LAYER_H

#include "ilayer.h"
#include "iparam.h"

template<MatMode mode, typename Dtype>
class ParamLayer : public ILayer<mode, Dtype>
{
public:
    ParamLayer(NNGraph<mode, Dtype>* _nn, IParam<mode, Dtype>* _param, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_nn, _properr) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
        this->param = _param;
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands) override
    {
        
    }
    
    IParam<mode, Dtype>* param;        
};

#endif