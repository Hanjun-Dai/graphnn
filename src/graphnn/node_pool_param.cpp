#include "node_pool_param.h"
#include "sparse_matrix.h"
#include "graph_data.h"
#include <iostream>

template<>
NodePoolParam<CPU, float>::NodePoolParam(std::string _name, PoolingOp _op, NodePoolType _npt) : 
								  IParam<CPU, float>(_name), op(_op), npt(_npt)
{
		assert(op != PoolingOp::MAX);
        cpu_weight = &weight;
}

template<>
NodePoolParam<CPU, double>::NodePoolParam(std::string _name, PoolingOp _op, NodePoolType _npt) : 
								  IParam<CPU, double>(_name), op(_op), npt(_npt)
{
		assert(op != PoolingOp::MAX);
        cpu_weight = &weight;
}

template<>
NodePoolParam<GPU, float>::NodePoolParam(std::string _name, PoolingOp _op, NodePoolType _npt) : 
								  IParam<GPU, float>(_name), op(_op), npt(_npt)
{
		assert(op != PoolingOp::MAX);
        cpu_weight = new SparseMat<CPU, float>();
}

template<>
NodePoolParam<GPU, double>::NodePoolParam(std::string _name, PoolingOp _op, NodePoolType _npt) : 
								  IParam<GPU, double>(_name), op(_op), npt(_npt)
{
		assert(op != PoolingOp::MAX);
        cpu_weight = new SparseMat<CPU, double>();
}

template<MatMode mode, typename Dtype>
void NodePoolParam<mode, Dtype>::InitializeBatch(GraphData<mode, Dtype>* g)
{		
		if (this->batch_prepared)
			return;
	   
        this->InitCPUWeight(g);
		if (mode == GPU)
            this->weight.CopyFrom(*(this->cpu_weight));
        
		this->batch_prepared = true;
}

template<MatMode mode, typename Dtype>
void NodePoolParam<mode, Dtype>::InitCPUWeight(GraphData<mode, Dtype>* g)
{
 		auto* graph = g->graph;
		if (npt == NodePoolType::E2N)
			cpu_weight->Resize(graph->num_nodes, g->edge_states->rows);
		else
			cpu_weight->Resize(graph->num_nodes, g->node_states->rows);
		
		cpu_weight->ResizeSp(graph->num_edges, graph->num_nodes + 1);
		
		int nnz = 0;
		auto& data = cpu_weight->data;
		Dtype scalar;
		for (int i = 0; i < graph->num_nodes; ++i)
		{
			data->ptr[i] = nnz;
			scalar = (op == PoolingOp::SUM ? 1.0 : 1.0 / graph->adj_list->head[i].size());
			auto& list = graph->adj_list->head[i];
			for (size_t j = 0; j < list.size(); ++j)
			{
				data->val[nnz] = scalar;
				data->col_idx[nnz] = npt == NodePoolType::E2N ?  list[j].first : list[j].second;
				nnz++;
			}
		}
		assert(nnz == graph->num_edges);
		data->ptr[graph->num_nodes] = nnz;       
}

template<MatMode mode, typename Dtype>
void NodePoolParam<mode, Dtype>::UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum) 
{
	
}

template<MatMode mode, typename Dtype>
void NodePoolParam<mode, Dtype>::Serialize(FILE* fid)
{
	IParam<mode, Dtype>::Serialize(fid);
	int i_op = (int)op;
	assert(fwrite(&i_op, sizeof(int), 1, fid) == 1);
	int i_npt = (int)npt;
	assert(fwrite(&i_npt, sizeof(int), 1, fid) == 1);
	weight.Serialize(fid);
}

template<MatMode mode, typename Dtype>
void NodePoolParam<mode, Dtype>::Deserialize(FILE* fid)
{
	IParam<mode, Dtype>::Deserialize(fid);
	int i_op, i_npt;
	assert(fread(&i_op, sizeof(int), 1, fid) == 1);
	op = (PoolingOp)i_op;
	assert(fread(&i_npt, sizeof(int), 1, fid) == 1);
	npt = (NodePoolType)i_npt;
	weight.Deserialize(fid);
}

template class NodePoolParam<CPU, double>;
template class NodePoolParam<CPU, float>;
template class NodePoolParam<GPU, double>;
template class NodePoolParam<GPU, float>;

// =============================== SubgraphPoolParam =========================================

template<MatMode mode, typename Dtype>
SubgraphPoolParam<mode, Dtype>::SubgraphPoolParam(std::string _name, PoolingOp _op, NodePoolType _npt) : 
								  		NodePoolParam<mode, Dtype>(_name, _op, _npt)
{											  
		assert(this->npt == NodePoolType::N2N);
}


template<MatMode mode, typename Dtype>
void SubgraphPoolParam<mode, Dtype>::InitCPUWeight(GraphData<mode, Dtype>* g)
{		
		auto* graph = g->graph;
		
		this->cpu_weight->Resize(graph->num_subgraph, g->node_states->rows);
		
		this->cpu_weight->ResizeSp(graph->num_nodes, graph->num_subgraph + 1);
		
		int nnz = 0;
		auto& data = this->cpu_weight->data;
		Dtype scalar;
		for (int i = 0; i < graph->num_subgraph; ++i)
		{
			data->ptr[i] = nnz;
			scalar = (this->op == PoolingOp::SUM ? 1.0 : 1.0 / graph->subgraph->head[i].size());
			auto& list = graph->subgraph->head[i];
			for (size_t j = 0; j < list.size(); ++j)
			{
				data->val[nnz] = scalar;
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