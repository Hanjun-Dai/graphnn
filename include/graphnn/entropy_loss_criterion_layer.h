#ifndef ENTROPY_LOSS_CRITERION_LAYER_H
#define ENTROPY_LOSS_CRITERION_LAYER_H

#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
class EntropyLossCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
        
			EntropyLossCriterionLayer(std::string _name, PropErr _properr = PropErr::T)
                : EntropyLossCriterionLayer<mode, Dtype>(_name, 1.0, _properr) {}
                
			EntropyLossCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T);
                        
            static std::string str_type()
            {
                return "EntropyLoss"; 
            }
            
            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override;
            
            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;
};

#endif