#include "msg_pass_param.h"
#include "sparse_matrix.h"
#include "dense_matrix.h"
#include "graph_data.h"
#include <iostream>

// =============================== IMessagePassParam =========================================

template<>
IMessagePassParam<CPU, float>::IMessagePassParam(std::string _name) : 
								  IParam<CPU, float>(_name)
{
        cpu_weight = &weight;
}

template<>
IMessagePassParam<CPU, double>::IMessagePassParam(std::string _name) :  
								  IParam<CPU, double>(_name)
{
        cpu_weight = &weight;
}

template<>
IMessagePassParam<GPU, float>::IMessagePassParam(std::string _name) : 
								  IParam<GPU, float>(_name)
{
        cpu_weight = new SparseMat<CPU, float>();
}

template<>
IMessagePassParam<GPU, double>::IMessagePassParam(std::string _name) : 
								  IParam<GPU, double>(_name)
{
        cpu_weight = new SparseMat<CPU, double>();
}

template<MatMode mode, typename Dtype>
void IMessagePassParam<mode, Dtype>::InitializeBatch(GraphData<mode, Dtype>* g, GraphAtt operand)
{		
		if (this->batch_prepared)
			return;
	   
        this->InitCPUWeight(g, operand);
		if (mode == GPU)
            this->weight.CopyFrom(*(this->cpu_weight));
        
		this->batch_prepared = true;
}

template<MatMode mode, typename Dtype>
void IMessagePassParam<mode, Dtype>::UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase)
{
        auto& prev_states = input->DenseDerived();
        
        output->SparseMM(this->weight, prev_states, Trans::N, Trans::N, 1.0, beta);                                                                       
}

template<MatMode mode, typename Dtype>
void IMessagePassParam<mode, Dtype>::UpdateGradInput(IMatrix<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta)
{
        auto& prev_grad = gradInput->DenseDerived();
        
        prev_grad.SparseMM(this->weight, *gradOutput, Trans::T, Trans::N, 1.0, beta);                                                             
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
void NodeCentricPoolParam<mode, Dtype>::InitCPUWeight(GraphData<mode, Dtype>* g, GraphAtt operand)
{
 		auto* graph = g->graph;
		if (operand == GraphAtt::EDGE)
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
				data->col_idx[nnz] = operand == GraphAtt::EDGE ?  list[j].first : list[j].second;
				nnz++;
			}
		}
		assert(nnz == graph->num_edges);
		data->ptr[graph->num_nodes] = nnz;       
}

template class NodeCentricPoolParam<CPU, double>;
template class NodeCentricPoolParam<CPU, float>;
template class NodeCentricPoolParam<GPU, double>;
template class NodeCentricPoolParam<GPU, float>;

// =============================== EdgePoolParam =========================================

template<MatMode mode, typename Dtype>
void EdgeCentricPoolParam<mode, Dtype>::InitCPUWeight(GraphData<mode, Dtype>* g, GraphAtt operand)
{
 		auto* graph = g->graph;
        int nnz = 0;
		if (operand == GraphAtt::EDGE)
        {
			this->cpu_weight->Resize(graph->num_edges, g->edge_states->rows);
            size_t cnt = 0;
            for (int i = 0; i < graph->num_nodes; ++i)
            {
                auto in_cnt = graph->in_edges->head[i].size();
                cnt += in_cnt * (in_cnt - 1); 
            }
            this->cpu_weight->ResizeSp(cnt, graph->num_edges + 1);            
            
            auto& data = this->cpu_weight->data;
            for (int i = 0; i < graph->num_edges; ++i)
            {
                data->ptr[i] = nnz;
                int node_from = graph->edge_list[i].first, node_to = graph->edge_list[i].second; 
                auto& list = graph->in_edges->head[node_from]; 
                for (size_t j = 0; j < list.size(); ++j)
                {
                    if (list[j].second == node_to)
                        continue; // the same edge in another direction
                    data->val[nnz] = 1.0;
                    data->col_idx[nnz] = list[j].first; // the edge index
                    nnz++;
                }
            }
            data->ptr[graph->num_edges] = nnz;
            assert(nnz == data->nnz);
            assert(data->nnz == cnt); 
        }
		else // NODE
        {
            this->cpu_weight->Resize(graph->num_edges, g->node_states->rows);
            this->cpu_weight->ResizeSp(graph->num_edges, graph->num_edges + 1);
            
		    auto& data = this->cpu_weight->data;
            for (int i = 0; i < graph->num_edges; ++i)
            {
                data->ptr[i] = nnz;
                data->val[nnz] = 1.0;
                data->col_idx[nnz] = graph->edge_list[i].first;
                nnz++;
            }
            data->ptr[graph->num_edges] = nnz;
            assert(nnz == data->nnz);     
        }
}

template class EdgeCentricPoolParam<CPU, double>;
template class EdgeCentricPoolParam<CPU, float>;
template class EdgeCentricPoolParam<GPU, double>;
template class EdgeCentricPoolParam<GPU, float>;

// =============================== SubgraphPoolParam =========================================

template<MatMode mode, typename Dtype>
void SubgraphPoolParam<mode, Dtype>::InitCPUWeight(GraphData<mode, Dtype>* g, GraphAtt operand)
{		
        assert(operand == GraphAtt::NODE);
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