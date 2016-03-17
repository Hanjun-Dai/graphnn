#include "graph_pool_param.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"

// =============================== NodeAvgPoolParam =========================================

template<MatMode mode, typename Dtype>
void NodeAvgPoolParam<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
        this->cpu_weight->Resize(graph->num_nodes, graph->num_nodes);
		this->cpu_weight->ResizeSp(graph->num_edges + graph->num_nodes, graph->num_nodes + 1);
		
		int nnz = 0;
		auto& data = this->cpu_weight->data;
		for (int i = 0; i < graph->num_nodes; ++i)
		{
			data->ptr[i] = nnz;
                        
			auto& list = graph->in_edges->head[i];
            data->val[nnz] = 1.0 / (list.size() + 1);
            data->col_idx[nnz] = i;
            nnz++;
			for (size_t j = 0; j < list.size(); ++j)
			{
				data->val[nnz] = 1.0 / (list.size() + 1);
				data->col_idx[nnz] = list[j].second;
				nnz++;
			}
		}
		assert(nnz == this->cpu_weight->data->nnz);
		data->ptr[graph->num_nodes] = nnz;
}

template class NodeAvgPoolParam<CPU, double>;
template class NodeAvgPoolParam<CPU, float>;
template class NodeAvgPoolParam<GPU, double>;
template class NodeAvgPoolParam<GPU, float>;

// =============================== NodeMaxPoolParam =========================================