#ifndef MVN_DIAG_NLL_CRITERION_LAYER_H
#define MVN_DIAG_NLL_CRITERION_LAYER_H

#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
class MVNDianNLLCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:

    MVNDianNLLCriterionLayer(std::string _name, PropErr _properr = PropErr::T)
        : MVNDianNLLCriterionLayer(_name, 1.0, _properr) {}
        
    MVNDianNLLCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T);

    static std::string str_type()
    {
        return "MVNDianNLLCriterion"; 
    }          
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override;
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;                 
};

#endif