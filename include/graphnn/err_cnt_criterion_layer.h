#ifndef ERR_CNT_CRITERION_LAYER_H
#define ERR_CNT_CRITERION_LAYER_H

#include "i_criterion_layer.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "loss_func.h"

template<MatMode mode, typename Dtype>
class ErrCntCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			ErrCntCriterionLayer(std::string _name)
                : ICriterionLayer<mode, Dtype>(_name, PropErr::N)
                {
                    
                }           
                     
            static std::string str_type()
            {
                return "ErrCnt"; 
            }
            
			virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
            {
                auto& pred = operands[0]->state->DenseDerived();
                auto& labels = operands[1]->state->SparseDerived();          
                this->loss = LossFunc<mode, Dtype>::GetErrCnt(pred, labels);
            }
            
            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
            {
                throw std::runtime_error("no grad in this layer");
            }                
                        
protected:
};

#endif