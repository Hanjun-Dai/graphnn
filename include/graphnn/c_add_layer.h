#ifndef C_ADD_LAYER_H
#define C_ADD_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class CAddLayer : public ILayer<mode, Dtype>
{
public:
    CAddLayer(std::string _name, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _properr) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        
    }    
};

#endif