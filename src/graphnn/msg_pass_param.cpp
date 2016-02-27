#include "msg_pass_param.h"
#include "sparse_matrix.h"
#include "dense_matrix.h"

// =============================== IMessagePassParam =========================================

template<>
IMessagePassParam<CPU, float>::IMessagePassParam(std::string _name) : 
								  IConstParam<CPU, float>(_name)
{
        cpu_weight = &weight;
}

template<>
IMessagePassParam<CPU, double>::IMessagePassParam(std::string _name) :  
								  IConstParam<CPU, double>(_name)
{
        cpu_weight = &weight;
}

template<>
IMessagePassParam<GPU, float>::IMessagePassParam(std::string _name) : 
								  IConstParam<GPU, float>(_name)
{
        cpu_weight = new SparseMat<CPU, float>();
}

template<>
IMessagePassParam<GPU, double>::IMessagePassParam(std::string _name) : 
								  IConstParam<GPU, double>(_name)
{
        cpu_weight = new SparseMat<CPU, double>();
}

template class IMessagePassParam<CPU, double>;
template class IMessagePassParam<CPU, float>;
template class IMessagePassParam<GPU, double>;
template class IMessagePassParam<GPU, float>;

// =============================== Node2NodePoolParam =========================================

template<MatMode mode, typename Dtype>
void Node2NodePoolParam<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
        this->cpu_weight->Resize(graph->num_nodes, graph->num_nodes);		
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
				data->col_idx[nnz] = list[j].second;
				nnz++;
			}
		}
		assert(nnz == graph->num_edges);
		data->ptr[graph->num_nodes] = nnz;
}

template class Node2NodePoolParam<CPU, double>;
template class Node2NodePoolParam<CPU, float>;
template class Node2NodePoolParam<GPU, double>;
template class Node2NodePoolParam<GPU, float>;

// =============================== Edge2NodePoolParam =========================================

template<MatMode mode, typename Dtype>
void Edge2NodePoolParam<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
		this->cpu_weight->Resize(graph->num_nodes, graph->num_edges);
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
				data->col_idx[nnz] = list[j].first;
				nnz++;
			}
		}
		assert(nnz == graph->num_edges);
		data->ptr[graph->num_nodes] = nnz;       
}

template class Edge2NodePoolParam<CPU, double>;
template class Edge2NodePoolParam<CPU, float>;
template class Edge2NodePoolParam<GPU, double>;
template class Edge2NodePoolParam<GPU, float>;

// =============================== Node2EdgePoolParam =========================================

template<MatMode mode, typename Dtype>
void Node2EdgePoolParam<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
        int nnz = 0;
        this->cpu_weight->Resize(graph->num_edges, graph->num_nodes);
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

template class Node2EdgePoolParam<CPU, double>;
template class Node2EdgePoolParam<CPU, float>;
template class Node2EdgePoolParam<GPU, double>;
template class Node2EdgePoolParam<GPU, float>;

// =============================== Edge2EdgePoolParam =========================================

template<MatMode mode, typename Dtype>
void Edge2EdgePoolParam<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
        int nnz = 0;
        this->cpu_weight->Resize(graph->num_edges, graph->num_edges);
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

template class Edge2EdgePoolParam<CPU, double>;
template class Edge2EdgePoolParam<CPU, float>;
template class Edge2EdgePoolParam<GPU, double>;
template class Edge2EdgePoolParam<GPU, float>;
// =============================== SubgraphPoolParam =========================================

template<MatMode mode, typename Dtype>
void SubgraphPoolParam<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{		
		this->cpu_weight->Resize(graph->num_subgraph, graph->num_nodes);		
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
