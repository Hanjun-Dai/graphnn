#ifndef GATHER_LAYER_H
#define GATHER_LAYER_H

#include "ilayer.h"
#include "iparam.h"

template<MatMode mode, typename Dtype>
class GatherLayer : public ILayer<mode, Dtype>
{
public:
	GatherLayer(std::string _name, GraphAtt _at, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _at, _properr)
    {
        this->graph_output = new GraphData<mode, Dtype>(DENSE);
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
    }
	
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override
    {
        auto* prev_output = prev_layer->graph_output;
        auto& prev_state = GetImatState(prev_output, this->at)->DenseDerived();
		auto& cur_output = GetImatState(this->graph_output, this->at)->DenseDerived();
		
		if (sv == SvType::WRITE2)
		{
			this->graph_output->graph = prev_output->graph;
            if (this->at == GraphAtt::NODE)
                this->graph_output->edge_states = prev_output->edge_states;
            else
                this->graph_output->node_states = prev_output->node_states;
                
			cur_output.CopyFrom(prev_state);
		} else if (sv == SvType::ADD2)
		{
			cur_output.Axpy(1.0, prev_state);
		}
    }
    
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
    {
        auto& cur_grad = GetImatState(this->graph_gradoutput, this->at)->DenseDerived();
		auto& prev_grad = GetImatState(prev_layer->graph_gradoutput, this->at)->DenseDerived();
		
		if (sv == SvType::WRITE2)
			prev_grad.CopyFrom(cur_grad);
		else 
			prev_grad.Axpy(1.0, cur_grad);
    }
};


template<MatMode mode, typename Dtype>
class NodeGatherLayer : public GatherLayer<mode, Dtype>
{
public:
    
    NodeGatherLayer(std::string _name, PropErr _properr = PropErr::T)
        : GatherLayer<mode, Dtype>(_name, GraphAtt::NODE, _properr) {}
}; 

template<MatMode mode, typename Dtype>
class EdgeGatherLayer : public GatherLayer<mode, Dtype>
{
public:
    
    EdgeGatherLayer(std::string _name, PropErr _properr = PropErr::T)
        : GatherLayer<mode, Dtype>(_name, GraphAtt::EDGE, _properr) {}
};

#endif