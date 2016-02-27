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
                this->graph_output = nullptr;
            }
                        			
			virtual Dtype IMatrix(IMatrix<mode, Dtype>* ground_truth) override
            {
                Dtype loss = 0.0;
                auto& node_diff = this->grad->DenseDerived();			
                node_diff.GeaM(1.0, Trans::N, this->state->DenseDerived(), -1.0, Trans::N, ground_truth->DenseDerived());
                loss += node_diff.Asum();
                        
		        return loss;                
            }
            
			virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx) override
            {
                throw std::runtime_error("not impltemented");
            }
};

#endif