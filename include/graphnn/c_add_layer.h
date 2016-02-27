#ifndef C_ADD_LAYER_H
#define C_ADD_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class CAddLayer : public ILayer<mode, Dtype>
{
public:
    CAddLayer(std::string _name, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _properr) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
    }
    
    static std::string str_type()
    {
        return "CAdd"; 
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        assert(operands.size());
        auto& cur_output = this->state->DenseDerived();
        
        for (size_t i = 0; i < operands.size(); ++i)
        {
            auto& prev_state = operands[i]->state->DenseDerived();
            
            if (i == 0)
                cur_output.CopyFrom(prev_state);
            else 
                cur_output.Axpy(1.0, prev_state);    
        }            
    }    
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
    {
        assert(cur_idx >= 0 && cur_idx < operands.size());
        auto& cur_grad = this->grad->DenseDerived();                
		auto& prev_grad = operands[cur_idx]->grad->DenseDerived();
		        
        if (beta == 0)
            prev_grad.CopyFrom(cur_grad);
        else
            prev_grad.Axpby(1.0, cur_grad, beta);
    }
};

#endif