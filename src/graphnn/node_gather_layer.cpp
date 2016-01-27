#include "node_gather_layer.h"
#include "dense_matrix.h"
#include "graph_data.h"
#include <iostream>

template<MatMode mode, typename Dtype>
NodeGatherLayer<mode, Dtype>::NodeGatherLayer(std::string _name, PropErr _properr) : 
								  ILayer<mode, Dtype>(_name, GraphAtt::NODE, _properr)
{
		this->graph_output = new GraphData<mode, Dtype>(DENSE);
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
}

template<MatMode mode, typename Dtype>
void NodeGatherLayer<mode, Dtype>::UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase)
{			
		auto* prev_output = prev_layer->graph_output;
		auto& cur_output = this->graph_output->node_states->DenseDerived();
		
		if (sv == SvType::WRITE2)
		{
			this->graph_output->graph = prev_output->graph;
			this->graph_output->edge_states = prev_output->edge_states;
			cur_output.CopyFrom(prev_output->node_states->DenseDerived());
		} else if (sv == SvType::ADD2)
		{
			cur_output.Axpy(1.0, prev_output->node_states->DenseDerived());
		}		
}

template<MatMode mode, typename Dtype>
void NodeGatherLayer<mode, Dtype>::BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv)
{
		auto& cur_grad = this->graph_gradoutput->node_states->DenseDerived();
		auto& prev_grad = prev_layer->graph_gradoutput->node_states->DenseDerived();
		
		if (sv == SvType::WRITE2)
			prev_grad.CopyFrom(cur_grad);
		else 
			prev_grad.Axpy(1.0, cur_grad);
}

template class NodeGatherLayer<CPU, float>;
template class NodeGatherLayer<CPU, double>;
template class NodeGatherLayer<GPU, float>;
template class NodeGatherLayer<GPU, double>;