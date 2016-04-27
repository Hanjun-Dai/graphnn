#ifndef MAX_ENTROPY_CRITERION_LAYER_H
#define MAX_ENTROPY_CRITERION_LAYER_H

#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
class MaxEntropyCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
        
			MaxEntropyCriterionLayer(std::string _name, PropErr _properr = PropErr::T)
                : MaxEntropyCriterionLayer<mode, Dtype>(_name, 1.0, _properr) {}
                
			MaxEntropyCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T);
                        
            static std::string str_type()
            {
                return "EntropyLoss"; 
            }
            
            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override;
            
            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;
};

#endif