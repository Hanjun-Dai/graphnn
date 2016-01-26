#ifndef COS_LAYER_H
#define COS_LAYER_H

#include "i_act_layer.h"

template<MatMode mode, typename Dtype>
class CosLayer : public IActLayer<mode, Dtype> 
{
public:
    CosLayer(std::string _name, WriteType _wt, GraphAtt _at, PropErr _properr = PropErr::T)
            : IActLayer<mode, Dtype>(_name, _wt, _at, _properr) 
    {
        assert(_wt == WriteType::OUTPLACE);
    }            

    virtual void Act(DenseMat<mode, Dtype>& prev_out, DenseMat<mode, Dtype>& cur_out) override
    {
        cur_out.Cos(prev_out);
    }
    
    virtual void Derivative(DenseMat<mode, Dtype>& dst, DenseMat<mode, Dtype>& prev_output, 
                            DenseMat<mode, Dtype>& cur_output, DenseMat<mode, Dtype>& cur_grad) override
    {
        dst.Sin(prev_output);
        dst.Scale(-1.0);
        dst.EleWiseMul(cur_grad);
    } 
};


#endif