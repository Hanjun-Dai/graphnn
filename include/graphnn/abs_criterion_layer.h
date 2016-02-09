#ifndef ABS_CRITERION_LAYER_H
#define ABS_CRITERION_LAYER_H


#include "icriterion_layer.h"

template<MatMode mode, typename Dtype>
class ABSCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			ABSCriterionLayer(std::string _name, PropErr _properr = PropErr::T)
                : ABSCriterionLayer(_name, 1.0, _properr) {}
                
            ABSCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T)
                : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr) 
            {
                this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
                this->graph_output = nullptr;
            }
            			
			virtual Dtype GetLoss(GraphData<mode, Dtype>* graph_truth) override
            {
                Dtype loss = 0.0;
                auto& node_diff = this->graph_gradoutput->node_states->DenseDerived();			
                node_diff.GeaM(1.0, Trans::N, this->graph_output->node_states->DenseDerived(), -1.0, Trans::N, graph_truth->node_states->DenseDerived());
                loss += node_diff.Asum();
                        
		        return loss;                
            }
            
			virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
            {
                throw "not impltemented";
            }
};

#endif