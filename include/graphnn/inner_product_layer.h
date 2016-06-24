#ifndef INNER_PRODUCT_LAYER_H
#define INNER_PRODUCT_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class InnerProductLayer : public ILayer<mode, Dtype>
{
public:
    InnerProductLayer(std::string _name, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _properr) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
    }
    
    static std::string str_type()
    {
        return "InnerProduct"; 
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        assert(operands.size() == 2);
        
        auto& cur_output = this->state->DenseDerived();
        buf.EleWiseMul(operands[0]->state->DenseDerived(), operands[1]->state->DenseDerived());
        ones.Resize(buf.cols, 1); 
        ones.Fill(1.0);
        cur_output.GeMM(buf, ones, Trans::N, Trans::N, 1.0, 0.0);
    }
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
    {
        assert(operands.size() == 2);
        
        auto& cur_grad = this->grad->DenseDerived();
        auto& prev_grad = operands[cur_idx]->grad->DenseDerived();
        auto& another_operand = operands[1 - cur_idx]->state->DenseDerived();

        buf.MulColVec(another_operand, cur_grad);
        if (beta == 0)
        	prev_grad.CopyFrom(buf);
        else
        	prev_grad.Axpby(1.0, buf, beta);
    }    
    
    DenseMat<mode, Dtype> buf, ones;
};


#endif