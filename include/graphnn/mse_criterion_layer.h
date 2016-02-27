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
            
			virtual Dtype GetLoss(IMatrix<mode, Dtype>* ground_truth) override
            {		
                auto& node_diff = this->grad->DenseDerived();						
                node_diff.GeaM(1.0, Trans::N, this->state->DenseDerived(), -1.0, Trans::N, ground_truth->DenseDerived());
                Dtype norm2 = node_diff.Norm2();
                Dtype loss = norm2 * norm2;
                
                if (this->properr == PropErr::T)
                    node_diff.Scale(2.0 * this->lambda / ground_truth->rows);   
		        return loss;
            }
            
			virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx) override
            {
                assert(operands.size() == 1 && cur_idx == 0);
                
                auto& cur_grad = this->grad->DenseDerived();
		        auto& prev_grad = operands[0]->grad->DenseDerived();
		
		        prev_grad.Axpy(1.0, cur_grad);	
            }
};

#endif