#ifndef CLASSNLL_CRITERION_LAYER_H
#define CLASSNLL_CRITERION_LAYER_H

#include "i_criterion_layer.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "loss_func.h"

template<MatMode mode, typename Dtype>
class ClassNLLCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			ClassNLLCriterionLayer(std::string _name, bool _need_softmax, PropErr _properr = PropErr::T)
                : ClassNLLCriterionLayer<mode, Dtype>(_name, _need_softmax, 1.0, _properr) {}
                
			ClassNLLCriterionLayer(std::string _name, bool _need_softmax, Dtype _lambda, PropErr _properr = PropErr::T)
				 : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr), need_softmax(_need_softmax)
            {
                this->grad = new DenseMat<mode, Dtype>();
            }
            
            static std::string str_type()
            {
                return "ClassNLL"; 
            }
            
            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
            {
                auto& top = this->grad->DenseDerived();
                top.CopyFrom(operands[0]->state->DenseDerived());
                if (need_softmax)
                    top.Softmax();
                auto& labels = operands[1]->state->SparseDerived();
                
                this->loss = LossFunc<mode, Dtype>::GetLogLoss(top, labels);
                
                if (need_softmax)
                {
                    top.Axpy(-1.0, labels); // calc grad
                    top.Scale(1.0 / top.rows); // normalize by batch size
                } else 
                {   
                    top.Inv();
                    top.EleWiseMul(labels);
                    top.Scale(-1.0 / top.rows); // normalize by batch size
                }
            }
            
            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
            {
                assert(operands.size() == 2 && cur_idx == 0);
                
                auto& prev_grad = operands[0]->grad->DenseDerived();
                auto& cur_grad = this->grad->DenseDerived();
                if (beta == 0)
                    prev_grad.CopyFrom(cur_grad);
                else
                    prev_grad.Axpby(1.0, cur_grad, beta);                
            }              
                        
protected:
            const bool need_softmax;
};

#endif