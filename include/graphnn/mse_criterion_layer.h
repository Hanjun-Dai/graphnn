#ifndef MSE_CRITERION_LAYER_H
#define MSE_CRITERION_LAYER_H

#include "icriterion_layer.h"
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
                this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
                this->graph_output = nullptr;
            }
            
			virtual Dtype GetLoss(GraphData<mode, Dtype>* graph_truth) override
            {		
		        int batch_size = graph_truth->node_states->rows;
                auto& node_diff = this->graph_gradoutput->node_states->DenseDerived();						
                node_diff.GeaM(1.0, Trans::N, this->graph_output->node_states->DenseDerived(), 1.0, Trans::N, graph_truth->node_states->DenseDerived());
                Dtype norm2 = node_diff.Norm2();
                Dtype loss = norm2 * norm2;
                
                if (this->properr == PropErr::T)
                    node_diff.Scale(2.0 * this->lambda / batch_size);   
		        return loss;
            }
            
			virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
            {
                auto& cur_grad = this->graph_gradoutput->node_states->DenseDerived();
		        auto& prev_grad = prev_layer->graph_gradoutput->node_states->DenseDerived();
		
		        if (sv == SvType::WRITE2)
                    prev_grad.CopyFrom(cur_grad);
		        else // add2
			        prev_grad.Axpy(1.0, cur_grad);	
            }
};

#endif