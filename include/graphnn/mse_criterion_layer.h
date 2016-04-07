#ifndef MSE_CRITERION_LAYER_H
#define MSE_CRITERION_LAYER_H

#include "i_criterion_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
class MSECriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			MSECriterionLayer(std::string _name, PropErr _properr = PropErr::T)
                : MSECriterionLayer(_name, 1.0, _properr) {}
                
			MSECriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T)
                : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr)
            {
                this->grad = new DenseMat<mode, Dtype>();
            }
            
			virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
            {		
                auto& node_diff = this->grad->DenseDerived();						
                node_diff.GeaM(1.0, Trans::N, operands[0]->state->DenseDerived(), -1.0, Trans::N, operands[1]->state->DenseDerived());
                Dtype norm2 = node_diff.Norm2();
                this->loss = norm2 * norm2;
                
                if (this->properr == PropErr::T)
                    node_diff.Scale(2.0 * this->lambda / operands[1]->state->rows);
            }
            
			virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
            {
                assert(operands.size() == 2 && cur_idx == 0);
                
                auto& cur_grad = this->grad->DenseDerived();
		        auto& prev_grad = operands[0]->grad->DenseDerived();
		
                if (beta == 0)
                    prev_grad.CopyFrom(cur_grad);
                else
                    prev_grad.Axpby(1.0, cur_grad, beta);	
            }
};

#endif