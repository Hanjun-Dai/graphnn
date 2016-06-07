#ifndef PARAM_LAYER_H
#define PARAM_LAYER_H

#include "i_layer.h"
#include "i_param.h"

template<MatMode mode, typename Dtype>
class ParamLayer : public ILayer<mode, Dtype>, public IParametric<mode, Dtype>
{
public:
    ParamLayer(std::string _name, std::vector< IParam<mode, Dtype>* > _params, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _properr), params(_params) 
    {
        this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
    }

    static std::string str_type()
    {
        return "Param"; 
    }
    
    virtual bool HasParam() override
    {
        return true;
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        assert(operands.size() == params.size());
        auto& cur_output = this->state->DenseDerived();
        for (size_t i = 0; i < operands.size(); ++i)
        {
            if (i == 0)
                params[i]->ResetOutput(operands[i]->state, &cur_output); 
            params[i]->UpdateOutput(operands[i]->state, &cur_output, i == 0 ? 0.0 : 1.0, phase);
        }
    }
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
    {
        auto& cur_grad = this->grad->DenseDerived();
        auto& prev_grad = operands[cur_idx]->grad->DenseDerived(); 
		
		params[cur_idx]->UpdateGradInput(&prev_grad, &cur_grad, beta);
    }    
    
    virtual void AccDeriv(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx) override
    {
        if (params[cur_idx]->IsDiff())
        {
            dynamic_cast<IDiffParam<mode, Dtype>*>(params[cur_idx])->AccDeriv(operands[cur_idx]->state,
                                                                              &this->grad->DenseDerived());
        }        
    }
    
    std::vector< IParam<mode, Dtype>* > params;
};

#endif