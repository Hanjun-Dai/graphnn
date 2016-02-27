#ifndef I_ACT_LAYER_H
#define I_ACT_LAYER_H

#include "i_layer.h"

enum class WriteType
{
	INPLACE = 0,
	OUTPLACE = 1	
};

template<MatMode mode, typename Dtype>
class IActLayer : public ILayer<mode, Dtype>
{
public:
	
	IActLayer(std::string _name, WriteType _wt, PropErr _properr = PropErr::T) 
        : ILayer<mode, Dtype>(_name, _properr), wt(_wt)
    {
        this->state = new DenseMat<mode, Dtype>();
        this->grad = new DenseMat<mode, Dtype>();
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        assert(operands.size() == 1);
        auto* prev_state = operands[0]->state;
        
        if (wt == WriteType::INPLACE)
            this->state = prev_state;
        else 
            this->state->DenseDerived().Resize(prev_state->rows, prev_state->cols);
        
        Act(prev_state->DenseDerived(), this->state->DenseDerived());             
    }
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
    {
        assert(operands.size() == 1 && cur_idx == 0);
        
        auto& prev_grad = operands[0]->grad->DenseDerived();			
		auto& cur_grad = this->grad->DenseDerived();            
        auto& prev_output = operands[0]->state->DenseDerived();
		auto& cur_output = this->state->DenseDerived();
                                
        Derivative(prev_grad, prev_output, cur_output, cur_grad, beta);        
    }
    
    virtual void Act(DenseMat<mode, Dtype>& prev_out, DenseMat<mode, Dtype>& cur_out) = 0;
    virtual void Derivative(DenseMat<mode, Dtype>& dst, DenseMat<mode, Dtype>& prev_output, 
                            DenseMat<mode, Dtype>& cur_output, DenseMat<mode, Dtype>& cur_grad, Dtype beta) = 0;
                                    
	WriteType wt;
};

#endif