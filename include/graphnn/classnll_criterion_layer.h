#ifndef CLASSNLL_CRITERION_LAYER_H
#define CLASSNLL_CRITERION_LAYER_H

#include "icriterion_layer.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"

template<MatMode mode, typename Dtype>
Dtype GetLogLoss(DenseMat<mode, Dtype>& pred, SparseMat<mode, Dtype>& label, DenseMat<mode, Dtype>& buf); 

template<MatMode mode, typename Dtype>
class ClassNLLCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			ClassNLLCriterionLayer(std::string _name, bool _need_softmax, PropErr _properr = PropErr::T);
			
			virtual Dtype GetLoss(GraphData<mode, Dtype>* graph_truth) override;
			virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override;
            
protected:
            const bool need_softmax;
            DenseMat<mode, Dtype> buf;
};

#endif