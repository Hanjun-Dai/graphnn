#ifndef C_MUL_LAYER_H
#define C_MUL_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class CMulLayer : public ILayer<mode, Dtype>
{
public:
    CMulLayer(std::string _name, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _properr) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
    }
    
    static std::string str_type()
    {
        return "CMul"; 
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        assert(operands.size() > 1);
        
        auto& cur_output = this->state->DenseDerived();
        cur_output.EleWiseMul(operands[0]->state->DenseDerived(), operands[1]->state->DenseDerived());
        
        for (size_t i = 2; i < operands.size(); ++i)
            cur_output.EleWiseMul(operands[i]->state->DenseDerived());        
    }
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
    {
        assert(operands.size() > 1);
        
        auto& cur_grad = this->grad->DenseDerived();
        
        auto& prev_grad = beta == 0 ? operands[cur_idx]->grad->DenseDerived() : buf;
        
        prev_grad.CopyFrom(cur_grad);
        
        for (size_t i = 0; i < operands.size(); ++i)
            if (i != cur_idx)
                prev_grad.EleWiseMul(operands[i]->state->DenseDerived());
                
        if (beta != 0)
            operands[cur_idx]->grad->DenseDerived().Axpby(1.0, prev_grad, beta);
    }    
    
    DenseMat<mode, Dtype> buf;
};

#endif