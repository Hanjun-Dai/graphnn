#ifndef EXPNLL_CRITERION_LAYER_H
#define EXPNLL_CRITERION_LAYER_H

#include "icriterion_layer.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"


template<MatMode mode, typename Dtype>
class ExpNLLCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
            ExpNLLCriterionLayer(std::string _name, PropErr _properr = PropErr::T)
                : ExpNLLCriterionLayer<mode, Dtype>(_name, 1.0, _properr) {}
            
            ExpNLLCriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T)
                : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr)
            {
                this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
		        this->graph_output = nullptr;
            }
			
			virtual Dtype GetLoss(GraphData<mode, Dtype>* graph_truth) override
            {                
			    int batch_size = graph_truth->node_states->rows; 
                auto& p = this->graph_output->node_states->DenseDerived();
                auto& y = graph_truth->node_states->DenseDerived();
                
                auto& grad = this->graph_gradoutput->node_states->DenseDerived();                
                grad.EleWiseDiv(y, p);
                
                // buf = log(p) + y / p
                buf.CopyFrom(p); 
                buf.Log();                  
                buf.Axpy(1.0, grad);
                
                Dtype loss = buf.Sum();
                
                grad.Scale(-this->lambda / batch_size);
                grad.Add(this->lambda / batch_size);
                grad.EleWiseDiv(p);
                
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
            
protected:
            DenseMat<mode, Dtype> buf;
};
#endif