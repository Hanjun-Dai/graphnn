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
            
			virtual Dtype GetLoss(IMatrix<mode, Dtype>* ground_truth) override
            {
                auto& pred = this->state->DenseDerived();
                auto& labels = ground_truth->SparseDerived();          
                Dtype loss = LossFunc<mode, Dtype>::GetErrCnt(pred, labels);
                return loss;
            }
            
protected:
};

#endif