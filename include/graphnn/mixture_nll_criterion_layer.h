#ifndef SUM_LOSS_CRITERION_LAYER_H
#define SUM_LOSS_CRITERION_LAYER_H

#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
class MixtureNLLCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:

    MixtureNLLCriterionLayer(std::string _name, PropErr _properr = PropErr::T)
        : MixtureNLLCriterionLayer(_name, 1.0, _properr) {}
        
    MixtureNLLCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T);
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override;
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;                      
};

#endif