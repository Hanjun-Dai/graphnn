#ifndef C_MUL_LAYER_H
#define C_MUL_LAYER_H

#include "ilayer.h"

template<MatMode mode, typename Dtype>
class CMulLayer : public ILayer<mode, Dtype>
{
public:
    CMulLayer(std::string _name, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _properr) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
    }
    
    static std::string str_type()
    {
        return "CMul"; 
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        
    }
};

#endif