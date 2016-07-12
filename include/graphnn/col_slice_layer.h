#ifndef COL_SLICE_LAYER_H
#define COL_SLICE_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class ColSliceLayer : public ILayer<mode, Dtype>
{
public:
    ColSliceLayer(std::string _name, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _properr) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
    }
    
    static std::string str_type()
    {
        return "ColSlice"; 
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {        
        assert(operands.size() >= 2);
        auto& cur_output = this->state->DenseDerived();
        
        if ((int)operands.size() == 2)
        {
            int col_idx = GetColIdx(operands[1]);
            
            auto& prev_output = operands[0]->state->DenseDerived();
            cur_output.GetColsFrom(prev_output, col_idx, 1);
        } else
        {
            throw std::runtime_error("only support single column selection");
        }       
    }    
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
    {
        assert(cur_idx == 0);
        auto& cur_grad = this->grad->DenseDerived();                        
		auto& prev_grad = operands[cur_idx]->grad->DenseDerived();
        
        if ((int)operands.size() == 2)
        {
            int col_idx = GetColIdx(operands[1]);
            prev_grad.SubmatAdd(0, col_idx, cur_grad, beta);
        } else
        {
            throw std::runtime_error("only support single column selection");
        }
    }
    
protected:

    inline int GetColIdx(ILayer<mode, Dtype>* op)
    {
        auto& col_selected = op->state->SparseDerived();
        assert(col_selected.data->nnz == 1);
        return col_selected.data->col_idx[0];
    }
};

#endif