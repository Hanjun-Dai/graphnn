#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "i_act_layer.h"

template<MatMode mode, typename Dtype>
class SoftmaxLayer; 

template<typename Dtype>
class SoftmaxLayer<CPU, Dtype> : public IActLayer<CPU, Dtype>
{
public:
    SoftmaxLayer(std::string _name, WriteType _wt = WriteType::OUTPLACE, PropErr _properr = PropErr::T)
            : IActLayer<CPU, Dtype>(_name, _wt, _properr) {}

    static std::string str_type()
    {
        return "Softmax"; 
    }
    
    virtual void Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out) override;
    
    virtual void Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                            DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad, Dtype beta) override;

    DenseMat<CPU, Dtype> buf;                            
}; 

template<typename Dtype>
class SoftmaxLayer<GPU, Dtype> : public IActLayer<GPU, Dtype>
{
public:
    SoftmaxLayer(std::string _name, WriteType _wt = WriteType::OUTPLACE, PropErr _properr = PropErr::T)
            : IActLayer<GPU, Dtype>(_name, _wt, _properr) {}

    static std::string str_type()
    {
        return "Softmax"; 
    }
    
    virtual void Act(DenseMat<GPU, Dtype>& prev_out, DenseMat<GPU, Dtype>& cur_out) override;
    
    virtual void Derivative(DenseMat<GPU, Dtype>& dst, DenseMat<GPU, Dtype>& prev_output, 
                            DenseMat<GPU, Dtype>& cur_output, DenseMat<GPU, Dtype>& cur_grad, Dtype beta) override;
    DenseMat<GPU, Dtype> buf;                           
};

#endif