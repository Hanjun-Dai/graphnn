#ifndef TANH_LAYER_H
#define TANH_LAYER_H

#include "i_act_layer.h"

template<MatMode mode, typename Dtype>
class TanhLayer; 

template<typename Dtype>
class TanhLayer<CPU, Dtype> : public IActLayer<CPU, Dtype> 
{
public:
    TanhLayer(std::string _name, WriteType _wt = WriteType::OUTPLACE, PropErr _properr = PropErr::T)
            : IActLayer<CPU, Dtype>(_name, _wt, _properr) {}            

    static std::string str_type()
    {
        return "Tanh"; 
    }

    virtual void Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out) override;
    
    virtual void Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                            DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad, Dtype beta) override; 
};

#endif