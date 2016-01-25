#include "node_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
NodeLayer<mode, Dtype>::NodeLayer(std::string _name, PropErr _properr) 
				: ILayer<mode, Dtype>(_name, _properr) 
{
		this->graph_output = new GraphData<mode, Dtype>(DENSE);
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);	
		params_map.clear();
}				

template<MatMode mode, typename Dtype>
void NodeLayer<mode, Dtype>::AddPrevLayerName(std::string _name)
{
		if (params_map.count(_name) == 0)
		{
			params_map[_name] = std::vector< IParam<mode, Dtype>* >();
		}
}

template<MatMode mode, typename Dtype>
void NodeLayer<mode, Dtype>::AddParam(std::string prev_layer_name, IParam<mode, Dtype>* _param)
{		
		AddPrevLayerName(prev_layer_name);
		params_map[prev_layer_name].push_back(_param);
}

template<MatMode mode, typename Dtype>
void NodeLayer<mode, Dtype>::UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase)
{
        //std::cerr << prev_layer->name << " " << this->name << std::endl;
		auto& cur_output = this->graph_output->node_states->DenseDerived();
				
		Dtype beta;				
		auto& params = params_map[prev_layer->name];
		auto* prev_output = prev_layer->graph_output;
		
		if (sv == SvType::WRITE2)
		{
			beta = 0.0;
			// we assume all the params have the same output size;
			cur_output.Resize(prev_output->graph->num_nodes, params[0]->OutSize());
			// we assume all the prev layers have the same graph structure;
			this->graph_output->graph = prev_output->graph;
			// we assume all the prev layers have the same edge states;
			this->graph_output->edge_states = prev_output->edge_states;
		} else
			beta = 1.0;
													
		assert(params.size() == 1);
		params[0]->InitializeBatch(prev_output);
		params[0]->UpdateOutput(prev_output->node_states, &cur_output, beta, phase);
}

template<MatMode mode, typename Dtype>
void NodeLayer<mode, Dtype>::BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv)
{
		auto& params = params_map[prev_layer->name];
		
		auto& cur_grad = this->graph_gradoutput->node_states->DenseDerived();
		auto& prev_grad = prev_layer->graph_gradoutput->node_states->DenseDerived();
		
		Dtype beta = 1.0;
		if (sv == SvType::WRITE2)
		{
			beta = 0.0;
			prev_grad.Resize(cur_grad.rows, params[0]->InSize());
		}
					
		// no grouping required
		assert(params.size() == 1);
		params[0]->UpdateGradInput(&prev_grad, &cur_grad, beta); 		
}

template<MatMode mode, typename Dtype>
void NodeLayer<mode, Dtype>::AccDeriv(ILayer<mode, Dtype>* prev_layer)
{
		auto& params = params_map[prev_layer->name];
		
		auto& cur_grad = this->graph_gradoutput->node_states->DenseDerived();
		
		params[0]->AccDeriv(prev_layer->graph_output->node_states, &cur_grad);			
}

template class NodeLayer<CPU, float>;
template class NodeLayer<CPU, double>;
template class NodeLayer<GPU, float>;
template class NodeLayer<GPU, double>;