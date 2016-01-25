#ifndef SIN_LAYER_H
#define SIN_LAYER_H

#include "i_act_layer.h"

template<MatMode mode, typename Dtype>
class SinLayer : public IActLayer<mode, Dtype> 
{
public:
    SinLayer(std::string _name, WriteType _wt, ActTarget _at, PropErr _properr = PropErr::T)
            : IActLayer<mode, Dtype>(_name, _wt, _at, _properr) 
    {
        assert(_wt == WriteType::OUTPLACE);
    }            

    virtual void Act(DenseMat<mode, Dtype>& prev_out, DenseMat<mode, Dtype>& cur_out) override
    {
        cur_out.Sin(prev_out);
    }
    
    virtual void Derivative(DenseMat<mode, Dtype>& dst, DenseMat<mode, Dtype>& prev_output, 
                            DenseMat<mode, Dtype>& cur_output, DenseMat<mode, Dtype>& cur_grad) override
    {
        dst.Cos(prev_output);
        dst.EleWiseMul(cur_grad);
    }                             
};

#endif