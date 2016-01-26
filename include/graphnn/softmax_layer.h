#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "i_act_layer.h"

template<MatMode mode, typename Dtype>
class SoftmaxLayer : public IActLayer<mode, Dtype>
{
public:
    SoftmaxLayer(std::string _name, WriteType _wt, GraphAtt _at, PropErr _properr = PropErr::T)
            : IActLayer<mode, Dtype>(_name, _wt, _at, _properr) {}

    virtual void Act(DenseMat<mode, Dtype>& prev_out, DenseMat<mode, Dtype>& cur_out) override
    {
        if (&cur_out != &prev_out)
            cur_out.CopyFrom(prev_out);
        cur_out.Softmax();
    }
    
    virtual void Derivative(DenseMat<mode, Dtype>& dst, DenseMat<mode, Dtype>& prev_output, 
                            DenseMat<mode, Dtype>& cur_output, DenseMat<mode, Dtype>& cur_grad) override 
                            {
                                assert(false); // not implemented yet
                            }                         
}; 

#endif