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
    
    static std::string str_type()
    {
        return "CAdd"; 
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        throw std::runtime_error("not implemented");
    }    
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx) override
    {
        throw std::runtime_error("not implemented");
    }
};

#endif