#ifndef ERR_CNT_CRITERION_LAYER_H
#define ERR_CNT_CRITERION_LAYER_H

#include "i_criterion_layer.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"


template<MatMode mode, typename Dtype>
Dtype GetErrCNT(DenseMat<mode, Dtype>& pred, SparseMat<mode, Dtype>& label, DenseMat<mode, Dtype>& buf); 

template<MatMode mode, typename Dtype>
class ErrCntCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			ErrCntCriterionLayer(std::string _name)
                : ICriterionLayer<mode, Dtype>(_name, PropErr::N)
                {
                    
                }                

			virtual Dtype GetLoss(IMatrix<mode, Dtype>* ground_truth) override
            {
                return 0;
            }
            
protected:
            DenseMat<mode, Dtype> buf;
};

#endif