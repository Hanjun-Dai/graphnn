#ifndef ABS_CRITERION_LAYER_H
#define ABS_CRITERION_LAYER_H

#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
class ABSCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			ABSCriterionLayer(std::string _name, PropErr _properr = PropErr::T)
                : ABSCriterionLayer(_name, 1.0, _properr) {}
                
            ABSCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T)
                : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr) 
            {
                this->grad = new DenseMat<mode, Dtype>();
            }
            
            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
            {
                assert(operands.size() == 2);
                
                auto& node_diff = this->grad->DenseDerived();			
                node_diff.GeaM(1.0, Trans::N, operands[0]->state->DenseDerived(), -1.0, Trans::N, operands[1]->state->DenseDerived());
                this->loss = node_diff.Asum();
            }
                                                			
			virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
            {
                throw std::runtime_error("not impltemented");
            }
};

#endif