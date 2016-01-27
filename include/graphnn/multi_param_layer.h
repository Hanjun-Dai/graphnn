#ifndef MULTI_PARAM_LAYER_H
#define MULTI_PARAM_LAYER_H

#include "ilayer.h"
#include "iparam.h"
#include <vector>
#include <map>

template<MatMode mode, typename Dtype>
class IMultiParamLayer : public ILayer<mode, Dtype>
{
public:
	typedef typename std::map<std::string, ILayer<mode, Dtype>* >::iterator layeriter;
	
	IMultiParamLayer(std::string _name, GraphAtt _at, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _at, _properr) 
    {
        this->graph_output = new GraphData<mode, Dtype>(DENSE);
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);	
		params_map.clear();
    }
	
	void AddParam(std::string prev_layer_name, IParam<mode, Dtype>* _param, GraphAtt _operand)
    {
        params_map[prev_layer_name] = std::make_pair(_param, _operand);
    }
	
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override
    {
        auto& cur_output = GetImatState(this->graph_output, this->at)->DenseDerived();

		Dtype beta;				
		auto* param = params_map[prev_layer->name].first;
        auto operand = params_map[prev_layer->name].second;
        
		auto* prev_output = prev_layer->graph_output;
		
		if (sv == SvType::WRITE2)
		{
			beta = 0.0;
			// we assume all the params have the same output size;
            if (param->OutSize()) // if we can know the outputsize ahead
            {
                auto num_states = this->at == GraphAtt::NODE ? prev_output->graph->num_nodes
                                                             : prev_output->graph->num_edges;
                                                             
                cur_output.Resize(num_states, param->OutSize());
            }
            // we assume all the prev layers have the same graph structure;
			this->graph_output->graph = prev_output->graph;

            if (this->at == GraphAtt::NODE)
                this->graph_output->edge_states = prev_output->edge_states; // edge state will remain the same
            else // EDGE
                this->graph_output->node_states = prev_output->node_states;
		} else
			beta = 1.0;
		
		param->InitializeBatch(prev_output, operand);
		param->UpdateOutput(GetImatState(prev_output, operand),  
                            &cur_output, beta, phase);
    }
    
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
    {
		auto* param = params_map[prev_layer->name].first;
        auto operand = params_map[prev_layer->name].second;
        
		auto& cur_grad = GetImatState(this->graph_gradoutput, this->at)->DenseDerived();
		
		Dtype beta = 1.0;
        auto& prev_grad = GetImatState(prev_layer->graph_gradoutput, operand)->DenseDerived(); 
		if (sv == SvType::WRITE2)
		{
			beta = 0.0;
            if (param->InSize()) // if we can know the inputsize ahead
                prev_grad.Resize(cur_grad.rows, param->InSize());
		}
		
		param->UpdateGradInput(&prev_grad, &cur_grad, beta);
    }
    
	virtual void AccDeriv(ILayer<mode, Dtype>* prev_layer) override
    {
		auto* param = params_map[prev_layer->name].first;
        auto operand = params_map[prev_layer->name].second;
		
		auto& cur_grad = GetImatState(this->graph_gradoutput, this->at)->DenseDerived();
		
		param->AccDeriv(GetImatState(prev_layer->graph_output, operand), &cur_grad);
    }
	
	std::map< std::string, std::pair< IParam<mode, Dtype>*, GraphAtt > > params_map;		
};

template<MatMode mode, typename Dtype>
class NodeLayer : public IMultiParamLayer<mode, Dtype>
{
public:
	NodeLayer(std::string _name, PropErr _properr = PropErr::T) : IMultiParamLayer<mode, Dtype>(_name, GraphAtt::NODE, _properr) {}	
};

template<MatMode mode, typename Dtype>
class EdgeLayer : public IMultiParamLayer<mode, Dtype>
{
public:
	EdgeLayer(std::string _name, PropErr _properr = PropErr::T) : IMultiParamLayer<mode, Dtype>(_name, GraphAtt::EDGE, _properr) {}	
};


#endif