#include "mse_criterion_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
MSECriterionLayer<mode, Dtype>::MSECriterionLayer(std::string _name, PropErr _properr)
								 : ICriterionLayer<mode, Dtype>(_name, _properr)
{
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
		this->graph_output = nullptr;
}

template<MatMode mode, typename Dtype>
Dtype MSECriterionLayer<mode, Dtype>::GetLoss(GraphData<mode, Dtype>* graph_truth)
{
		Dtype loss = 0.0;		
		if (graph_truth->node_states->count)
		{
			int batch_size = graph_truth->node_states->rows;
			auto& node_diff = this->graph_gradoutput->node_states->DenseDerived();						
			node_diff.GeaM(2.0 / batch_size, Trans::N, this->graph_output->node_states->DenseDerived(), -2.0 / batch_size, Trans::N, graph_truth->node_states->DenseDerived());
			Dtype norm2 = node_diff.Norm2() / 2 * batch_size;
			loss += norm2 * norm2;
		}
		
		if (graph_truth->edge_states->count)
		{
			throw "not impltemented";
		}
		
		return loss;
}

template<MatMode mode, typename Dtype>
void MSECriterionLayer<mode, Dtype>::BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv)
{		
		auto& cur_grad = this->graph_gradoutput->node_states->DenseDerived();
		auto& prev_grad = prev_layer->graph_gradoutput->node_states->DenseDerived();
		
		if (sv == SvType::WRITE2)
			prev_grad.CopyFrom(cur_grad);
		else // add2
			prev_grad.Axpy(1.0, cur_grad);
		
}

template class MSECriterionLayer<CPU, float>;
template class MSECriterionLayer<CPU, double>;
template class MSECriterionLayer<GPU, float>;
template class MSECriterionLayer<GPU, double>;