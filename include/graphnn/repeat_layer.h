#ifndef REPEAT_LAYER_H
#define REPEAT_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class RepeatLayer : public ILayer<mode, Dtype>
{
public:
    RepeatLayer(std::string _name, size_t _rep_cnt, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _properr), rep_cnt(_rep_cnt) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
    }
    
    static std::string str_type()
    {
        return "Repeat"; 
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        assert(operands.size() == 1);
        auto& cur_output = this->state->DenseDerived();
        auto& operand = operands[0]->state->DenseDerived();

        cur_output.Repmat(operand, this->rep_cnt, 1);
    }    
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
    {
        auto& cur_grad = this->grad->DenseDerived();
        auto& prev_grad = operands[cur_idx]->grad->DenseDerived();

        buf.RowSum(cur_grad);
        prev_grad.Axpby(1.0, buf, beta);
    }

    DenseMat<mode, Dtype> buf;
    size_t rep_cnt;
};

#endif