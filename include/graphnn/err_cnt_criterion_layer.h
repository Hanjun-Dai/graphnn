#ifndef ERR_CNT_CRITERION_LAYER_H
#define ERR_CNT_CRITERION_LAYER_H

#include "icriterion_layer.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"


template<MatMode mode, typename Dtype>
Dtype GetErrCNT(DenseMat<mode, Dtype>& pred, SparseMat<mode, Dtype>& label, DenseMat<mode, Dtype>& buf); 

template<MatMode mode, typename Dtype>
class ErrCntCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			ErrCntCriterionLayer(std::string _name);
			
			virtual Dtype GetLoss(GraphData<mode, Dtype>* graph_truth) override;
			virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override {}
            
protected:
            DenseMat<mode, Dtype> buf;
};

#endif