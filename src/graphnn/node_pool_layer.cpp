#include "node_pool_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
NodePoolLayer<mode, Dtype>::NodePoolLayer(std::string _name, NodePoolParam<mode, Dtype>* _param, PropErr _properr) : 
								  ILayer<mode, Dtype>(_name, _properr), param(_param)
{
		this->graph_output = new GraphData<mode, Dtype>(DENSE);
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
}

template<MatMode mode, typename Dtype>
void NodePoolLayer<mode, Dtype>::UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase)
{
        //std::cerr << prev_layer->name << " " << this->name << std::endl;
		assert(sv == SvType::WRITE2);
		auto* prev_output = prev_layer->graph_output;
		
		param->InitializeBatch(prev_output);
		
		auto& cur_output = this->graph_output->node_states->DenseDerived();
		auto& prev_states = param->npt == NodePoolType::E2N ? prev_output->edge_states->DenseDerived() : prev_output->node_states->DenseDerived();
		
        
		cur_output.SparseMM(param->weight, prev_states, Trans::N, Trans::N, 1.0, 0.0);
        //std::cerr << cur_output.Asum() << std::endl;                 		
		this->graph_output->graph = prev_output->graph;
		this->graph_output->edge_states = prev_output->edge_states;
}

template<MatMode mode, typename Dtype>
void NodePoolLayer<mode, Dtype>::BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv)
{
        auto& cur_grad = this->graph_gradoutput->node_states->DenseDerived();
        auto* prev_mat = param->npt == NodePoolType::E2N ? prev_layer->graph_gradoutput->edge_states : prev_layer->graph_gradoutput->node_states;
        auto& prev_grad = prev_mat->DenseDerived();    
        prev_grad.SparseMM(param->weight, cur_grad, Trans::T, Trans::N, 1.0, sv == SvType::WRITE2 ? 0.0 : 1.0);
}

template class NodePoolLayer<CPU, float>;
template class NodePoolLayer<CPU, double>;
template class NodePoolLayer<GPU, float>;
template class NodePoolLayer<GPU, double>;