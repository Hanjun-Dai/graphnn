#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "i_act_layer.h"

template<MatMode mode, typename Dtype>
class ReLULayer : public IActLayer<mode, Dtype> 
{
public:
    
    ReLULayer(std::string _name, WriteType _wt = WriteType::INPLACE, PropErr _properr = PropErr::T)
            : IActLayer<mode, Dtype>(_name, _wt, _properr) {}
};

#endif