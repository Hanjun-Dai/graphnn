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
	
	IMultiParamLayer(std::string _name, GraphAtt _at, PropErr _properr = PropErr::T) : ILayer<mode, Dtype>(_name, _at, _properr) 
    {
        this->graph_output = new GraphData<mode, Dtype>(DENSE);
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);	
		params_map.clear();
    }
	
	void AddParam(std::string prev_layer_name, IParam<mode, Dtype>* _param)
    {
        params_map[prev_layer_name] = _param;
    }
	
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override
    {
        auto& cur_output = GetImatState(this->graph_output, this->at)->DenseDerived();

		Dtype beta;				
		auto* param = params_map[prev_layer->name];
		auto* prev_output = prev_layer->graph_output;
		
		if (sv == SvType::WRITE2)
		{
			beta = 0.0;
			// we assume all the params have the same output size;
            if (param->OutSize()) // if we can know the outputsize ahead
                cur_output.Resize(prev_output->graph->num_nodes, param->OutSize());
			// we assume all the prev layers have the same graph structure;
			this->graph_output->graph = prev_output->graph;
			// we assume all the prev layers have the same edge states;
            if (param->operand == GraphAtt::NODE)
                this->graph_output->edge_states = prev_output->edge_states; // edge state will remain the same
            else // EDGE
                this->graph_output->node_states = prev_output->node_states;
		} else
			beta = 1.0;
											
		param->InitializeBatch(prev_output);
		param->UpdateOutput(prev_output, &cur_output, beta, phase);
    }
    
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
    {
        auto* param = params_map[prev_layer->name];
		
		auto& cur_grad = GetImatState(this->graph_gradoutput, this->att)->DenseDerived();
		
		Dtype beta = 1.0;
		if (sv == SvType::WRITE2)
		{
			beta = 0.0;
            if (param->InSize()) // if we can know the inputsize ahead
            {
                auto& prev_grad = GetImatState(prev_layer->graph_gradoutput, param->operand); 
                prev_grad.Resize(cur_grad.rows, param->InSize());
            }
		}
		
		param->UpdateGradInput(prev_layer->graph_gradoutput, &cur_grad, beta); 		
    }
    
	virtual void AccDeriv(ILayer<mode, Dtype>* prev_layer) override
    {
        auto* param = params_map[prev_layer->name];
		
		auto& cur_grad = GetImatState(this->graph_gradoutput, this->at)->DenseDerived();
		
		param->AccDeriv(prev_layer->graph_output, &cur_grad);
    }
	
	std::map< std::string, IParam<mode, Dtype>* > params_map;		
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