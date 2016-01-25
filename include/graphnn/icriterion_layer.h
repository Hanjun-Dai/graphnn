#ifndef ICRITERION_LAYER_H
#define ICRITERION_LAYER_H

#include "ilayer.h"

template<MatMode mode, typename Dtype>
class ICriterionLayer : public ILayer<mode, Dtype>
{
public:
		ICriterionLayer(std::string _name, PropErr _properr = PropErr::T) : ILayer<mode, Dtype>(_name, _properr) {}
		
		virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override
		{
				assert(sv == SvType::WRITE2);
				
				this->graph_output = prev_layer->graph_output;
		}
		
		virtual Dtype GetLoss(GraphData<mode, Dtype>* graph_truth) = 0;				
};

#endif