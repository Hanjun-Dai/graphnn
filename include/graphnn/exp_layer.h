#ifndef EXP_LAYER_H
#define EXP_LAYER_H

#include "i_act_layer.h"

template<MatMode mode, typename Dtype>
class ExpLayer : public IActLayer<mode, Dtype> 
{
public:
    
    ExpLayer(std::string _name, WriteType _wt = WriteType::OUTPLACE, PropErr _properr = PropErr::T)
            : IActLayer<mode, Dtype>(_name, _wt, _properr) {}

    static std::string str_type()
    {
        return "Exp"; 
    }

    virtual void Act(DenseMat<mode, Dtype>& prev_out, DenseMat<mode, Dtype>& cur_out) override
    {
        cur_out.Exp(prev_out);
    }
    
    virtual void Derivative(DenseMat<mode, Dtype>& dst, DenseMat<mode, Dtype>& prev_output, 
                            DenseMat<mode, Dtype>& cur_output, DenseMat<mode, Dtype>& cur_grad, Dtype beta) override
    {
        assert(beta == 0);
        
        dst.CopyFrom(cur_grad);
        dst.EleWiseMul(cur_output);
    }                                 
};

#endif