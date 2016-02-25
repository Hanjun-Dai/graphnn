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
    
    virtual void Act(DenseMat<mode, Dtype>& prev_out, DenseMat<mode, Dtype>& cur_out) = 0;
    virtual void Derivative(DenseMat<mode, Dtype>& dst, DenseMat<mode, Dtype>& prev_output, 
                            DenseMat<mode, Dtype>& cur_output, DenseMat<mode, Dtype>& cur_grad) = 0;
                                    
	WriteType wt;
};

#endif