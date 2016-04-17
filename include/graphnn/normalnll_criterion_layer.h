#ifndef NORMAL_NLL_CRITERION_LAYER_H
#define NORMAL_NLL_CRITERION_LAYER_H

#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
class NormalNLLCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:

    NormalNLLCriterionLayer(std::string _name, PropErr _properr = PropErr::T)
        : NormalNLLCriterionLayer(_name, 1.0, _properr) {}
        
    NormalNLLCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T);    
        
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override;
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;                 
};

#endif