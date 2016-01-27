#include "msg_pass_param.h"
#include "sparse_matrix.h"
#include "graph_data.h"
#include <iostream>

// =============================== IMessagePassParam =========================================

template<>
IMessagePassParam<CPU, float>::IMessagePassParam(std::string _name, GraphAtt _operand) : 
								  IParam<CPU, float>(_name, _operand)
{
        cpu_weight = &weight;
}

template<>
IMessagePassParam<CPU, double>::IMessagePassParam(std::string _name, GraphAtt _operand) :  
								  IParam<CPU, double>(_name, _operand)
{
        cpu_weight = &weight;
}

template<>
IMessagePassParam<GPU, float>::IMessagePassParam(std::string _name, GraphAtt _operand) : 
								  IParam<GPU, float>(_name, _operand)
{
        cpu_weight = new SparseMat<CPU, float>();
}

template<>
IMessagePassParam<GPU, double>::IMessagePassParam(std::string _name, GraphAtt _operand) : 
								  IParam<GPU, double>(_name, _operand)
{
        cpu_weight = new SparseMat<CPU, double>();
}

template<MatMode mode, typename Dtype>
void IMessagePassParam<mode, Dtype>::InitializeBatch(GraphData<mode, Dtype>* g)
{		
		if (this->batch_prepared)
			return;
	   
        this->InitCPUWeight(g);
		if (mode == GPU)
            this->weight.CopyFrom(*(this->cpu_weight));
        
		this->batch_prepared = true;
}

template<MatMode mode, typename Dtype>
void IMessagePassParam<mode, Dtype>::UpdateOutput(GraphData<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase)
{
    
}

template<MatMode mode, typename Dtype>
void IMessagePassParam<mode, Dtype>::UpdateGradInput(GraphData<mode, Dtype>* gradInput_graph, DenseMat<mode, Dtype>* gradOutput, Dtype beta)
{
    
}

template<MatMode mode, typename Dtype>
void IMessagePassParam<mode, Dtype>::Serialize(FILE* fid)
{
	IParam<mode, Dtype>::Serialize(fid);	
	weight.Serialize(fid);
}

template<MatMode mode, typename Dtype>
void IMessagePassParam<mode, Dtype>::Deserialize(FILE* fid)
{
	IParam<mode, Dtype>::Deserialize(fid);
	weight.Deserialize(fid);
}

template class IMessagePassParam<CPU, double>;
template class IMessagePassParam<CPU, float>;
template class IMessagePassParam<GPU, double>;
template class IMessagePassParam<GPU, float>;

// =============================== NodePoolParam =========================================

template<MatMode mode, typename Dtype>
void NodePoolParam<mode, Dtype>::InitCPUWeight(GraphData<mode, Dtype>* g)
{
 		auto* graph = g->graph;
		if (this->operand == GraphAtt::EDGE)
			this->cpu_weight->Resize(graph->num_nodes, g->edge_states->rows);
		else // NODE
			this->cpu_weight->Resize(graph->num_nodes, g->node_states->rows);
		
		this->cpu_weight->ResizeSp(graph->num_edges, graph->num_nodes + 1);
		
		int nnz = 0;
		auto& data = this->cpu_weight->data;		
		for (int i = 0; i < graph->num_nodes; ++i)
		{
			data->ptr[i] = nnz;
			auto& list = graph->in_edges->head[i];
			for (size_t j = 0; j < list.size(); ++j)
			{
				data->val[nnz] = 1.0;
				data->col_idx[nnz] = this->operand == GraphAtt::EDGE ?  list[j].first : list[j].second;
				nnz++;
			}
		}
		assert(nnz == graph->num_edges);
		data->ptr[graph->num_nodes] = nnz;       
}

template class NodePoolParam<CPU, double>;
template class NodePoolParam<CPU, float>;
template class NodePoolParam<GPU, double>;
template class NodePoolParam<GPU, float>;

// =============================== SubgraphPoolParam =========================================

template<MatMode mode, typename Dtype>
void SubgraphPoolParam<mode, Dtype>::InitCPUWeight(GraphData<mode, Dtype>* g)
{		
		auto* graph = g->graph;
		
		this->cpu_weight->Resize(graph->num_subgraph, g->node_states->rows);
		
		this->cpu_weight->ResizeSp(graph->num_nodes, graph->num_subgraph + 1);
		
		int nnz = 0;
		auto& data = this->cpu_weight->data;
		for (int i = 0; i < graph->num_subgraph; ++i)
		{
			data->ptr[i] = nnz;
			auto& list = graph->subgraph->head[i];
			for (size_t j = 0; j < list.size(); ++j)
			{
				data->val[nnz] = 1.0;
				data->col_idx[nnz] = list[j];
				nnz++;
			}
		}
		assert(nnz == graph->num_nodes);
		data->ptr[graph->num_subgraph] = nnz;
}


template class SubgraphPoolParam<CPU, double>;
template class SubgraphPoolParam<CPU, float>;
template class SubgraphPoolParam<GPU, double>;
template class SubgraphPoolParam<GPU, float>;