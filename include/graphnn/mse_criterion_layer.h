#ifndef MSE_CRITERION_LAYER_H
#define MSE_CRITERION_LAYER_H


#include "icriterion_layer.h"

template<MatMode mode, typename Dtype>
class MSECriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			MSECriterionLayer(std::string _name, PropErr _properr = PropErr::T);
			
			virtual Dtype GetLoss(GraphData<mode, Dtype>* graph_truth) override;
			virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override;
};

#endif