#ifndef LOG_LAYER_H
#define LOG_LAYER_H

#include "i_act_layer.h"

template<MatMode mode, typename Dtype>
class LogLayer : public IActLayer<mode, Dtype>
{
public:

        LogLayer(std::string _name, PropErr _properr = PropErr::T)
            : IActLayer<mode, Dtype>(_name, WriteType::OUTPLACE, _properr) {}
            
        static std::string str_type()
        {
            return "Log"; 
        }

        virtual void Act(DenseMat<mode, Dtype>& prev_out, DenseMat<mode, Dtype>& cur_out) override
        {
            cur_out.Log(prev_out);
        }
    
        virtual void Derivative(DenseMat<mode, Dtype>& dst, DenseMat<mode, Dtype>& prev_output, 
                                DenseMat<mode, Dtype>& cur_output, DenseMat<mode, Dtype>& cur_grad, Dtype beta) override
        {
            buf.CopyFrom(prev_output);
            buf.Inv();
            buf.EleWiseMul(cur_grad);
            
            dst.Axpby(1.0, buf, beta);
        }                                               
        
protected:

        DenseMat<mode, Dtype> buf;        
};

#endif