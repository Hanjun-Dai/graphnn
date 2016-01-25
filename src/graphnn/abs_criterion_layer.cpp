#include "abs_criterion_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
ABSCriterionLayer<mode, Dtype>::ABSCriterionLayer(std::string _name, PropErr _properr)
								 : ICriterionLayer<mode, Dtype>(_name, _properr)
{
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
		this->graph_output = nullptr;
}

template<MatMode mode, typename Dtype>
Dtype ABSCriterionLayer<mode, Dtype>::GetLoss(GraphData<mode, Dtype>* graph_truth)
{
		Dtype loss = 0.0;
		if (graph_truth->node_states->count)
		{
			auto& node_diff = this->graph_gradoutput->node_states->DenseDerived();			
			node_diff.GeaM(1.0, Trans::N, this->graph_output->node_states->DenseDerived(), -1.0, Trans::N, graph_truth->node_states->DenseDerived());
			loss += node_diff.Asum();
		}
		
		if (graph_truth->edge_states->count)
		{
			throw "not impltemented";
		}
		
		return loss;
}

template<MatMode mode, typename Dtype>
void ABSCriterionLayer<mode, Dtype>::BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv)
{
		throw "not impltemented";
}

template class ABSCriterionLayer<CPU, float>;
template class ABSCriterionLayer<CPU, double>;
template class ABSCriterionLayer<GPU, float>;
template class ABSCriterionLayer<GPU, double>;