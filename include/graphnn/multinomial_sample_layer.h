#ifndef MULTINOMIAL_SAMPLE_LAYER_H
#define MULTINOMIAL_SAMPLE_LAYER_H

#include "i_act_layer.h"
#include<random>
#include<chrono>

template<MatMode mode, typename Dtype>
class MultinomialSampleLayer;

enum class SampleType
{
	MAX = 0,
	STOCHASTIC = 1	
};

template<typename Dtype>
class MultinomialSampleLayer<CPU, Dtype> : public IActLayer<CPU, Dtype> 
{
public:
    
    MultinomialSampleLayer(std::string _name, SampleType _st, PropErr _properr = PropErr::T)
            : IActLayer<CPU, Dtype>(_name, WriteType::OUTPLACE, _properr), st(_st) {}

    static std::string str_type()
    {
        return "MultinomialSample"; 
    }            

    virtual void Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out) override;
    
    virtual void Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                            DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad, Dtype beta) override;     

    SampleType st;

};


#endif