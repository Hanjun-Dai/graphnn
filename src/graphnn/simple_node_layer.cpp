#include "simple_node_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
SimpleNodeLayer<mode, Dtype>::SimpleNodeLayer(std::string _name, IParam<mode, Dtype>* _param, PropErr _properr) 
				: ILayer<mode, Dtype>(_name, _properr), param(_param) 
{
		this->graph_output = new GraphData<mode, Dtype>(DENSE);
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
}

template<MatMode mode, typename Dtype>
void SimpleNodeLayer<mode, Dtype>::UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase)
{
        assert(sv == SvType::WRITE2); 
		auto& cur_output = this->graph_output->node_states->DenseDerived();
				
		auto* prev_output = prev_layer->graph_output;
		
		cur_output.Resize(prev_output->graph->num_nodes, param->OutSize());        		
		this->graph_output->graph = prev_output->graph;		
		this->graph_output->edge_states = prev_output->edge_states;
													
		
		param->InitializeBatch(prev_output);
        
		param->UpdateOutput(prev_output->node_states, &cur_output, 0.0, phase);
}

template<MatMode mode, typename Dtype>
void SimpleNodeLayer<mode, Dtype>::BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv)
{		
		auto& cur_grad = this->graph_gradoutput->node_states->DenseDerived();
		auto& prev_grad = prev_layer->graph_gradoutput->node_states->DenseDerived();
		
		Dtype beta = 1.0;
		if (sv == SvType::WRITE2)
		{
			beta = 0.0;
			prev_grad.Resize(cur_grad.rows, param->InSize());
		}
        
		param->UpdateGradInput(&prev_grad, &cur_grad, beta);
}

template<MatMode mode, typename Dtype>
void SimpleNodeLayer<mode, Dtype>::AccDeriv(ILayer<mode, Dtype>* prev_layer)
{		
		auto& cur_grad = this->graph_gradoutput->node_states->DenseDerived();			
		param->AccDeriv(prev_layer->graph_output->node_states, &cur_grad);	
}

template class SimpleNodeLayer<CPU, float>;
template class SimpleNodeLayer<CPU, double>;
template class SimpleNodeLayer<GPU, float>;
template class SimpleNodeLayer<GPU, double>;