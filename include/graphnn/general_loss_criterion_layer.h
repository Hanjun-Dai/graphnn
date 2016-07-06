#ifndef GENERAL_LOSS_CRITERION_LAYER_H
#define GENERAL_LOSS_CRITERION_LAYER_H

#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
class GeneralLossCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
            GeneralLossCriterionLayer(std::string _name, PropErr _properr = PropErr::T)
                : GeneralLossCriterionLayer(_name, 1.0, _properr) {}
                
			GeneralLossCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T)
                : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr)
            {
                
            }
            
            static std::string str_type()
            {
                return "GeneralLossCriterion"; 
            }
            
			virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
            {		
                assert(operands.size() == 1);
                this->loss = operands[0]->state->DenseDerived().Sum() * this->lambda;
            }
            
			virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
            {
                assert(operands.size() == 1 && cur_idx == 0);
                auto& prev_grad = operands[0]->grad->DenseDerived();
                
                if (beta == 0)
                    prev_grad.Fill(this->lambda / operands[0]->state->rows);
                else{
                    prev_grad.Scale(beta);
                    prev_grad.Add(this->lambda / operands[0]->state->rows);
                }
            }    
};

#endif