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

template<typename Dtype>
void NodeMaxPoolParam<CPU, Dtype>::UpdateOutput(IMatrix<CPU, Dtype>* input, DenseMat<CPU, Dtype>* output, Dtype beta, Phase phase)
{
        if (max_index.size() < output->count)
            max_index.resize(output->count);
        Dtype* input_feature = input->DenseDerived().data;
        size_t cur_offset, neighbor_offset;
        for (size_t i = 0; i < graph->num_nodes; ++i)
        {
            auto& list = graph->in_edges->head[i];
            for (size_t j = 0; j < output->cols; ++j)
            {
                cur_offset = i * input->cols + j;
                Dtype cur_best = input_feature[cur_offset];
                max_index[cur_offset] = cur_offset;
                for (size_t k = 0; k < list.size(); ++k)
                {
                    neighbor_offset = list[k].second * input->cols + j;
                    if (input_feature[neighbor_offset] > cur_best)
                    {
                        cur_best = input_feature[neighbor_offset];
                        max_index[cur_offset] = neighbor_offset;
                    }                    
                }
                output->data[cur_offset] = output->data[cur_offset] * beta + cur_best;
            }            
        }
}

template<typename Dtype>
void NodeMaxPoolParam<CPU, Dtype>::UpdateGradInput(DenseMat<CPU, Dtype>* gradInput, DenseMat<CPU, Dtype>* gradOutput, Dtype beta)
{
        auto& prev_grad = gradInput->DenseDerived();
        
        prev_grad.Scale(beta);
        size_t offset;
        for (size_t i = 0; i < prev_grad.rows; ++i)
            for (size_t j = 0; j < prev_grad.cols; ++j)
            {
                offset = i * prev_grad.cols + j;
                prev_grad.data[max_index[offset]] += gradOutput->data[offset];
            }
}

template class NodeMaxPoolParam<CPU, double>;
template class NodeMaxPoolParam<CPU, float>;