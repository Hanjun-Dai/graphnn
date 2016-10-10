#ifndef TRANSPOSE_LAYER_H
#define TRANSPOSE_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class TransposeLayer : public ILayer<mode, Dtype>
{
public:
    TransposeLayer(std::string _name, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _properr) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
    }
    
    static std::string str_type()
    {
        return "Transpose"; 
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        assert(operands.size() == 1);
        auto& cur_output = this->state->DenseDerived();
        auto& operand = operands[0]->state->DenseDerived();

        assert(operand.rows == 1 || operand.cols == 1);

        cur_output.CopyFrom(operand);
        cur_output.rows = operand.cols;
        cur_output.cols = operand.rows;
    }    
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
    {
        auto& cur_grad = this->grad->DenseDerived();
        auto& prev_grad = operands[cur_idx]->grad->DenseDerived();

        std::swap(cur_grad.rows, cur_grad.cols);
        prev_grad.Axpby(1.0, cur_grad, beta);
        std::swap(cur_grad.rows, cur_grad.cols);
    }
};

#endif