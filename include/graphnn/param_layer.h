#ifndef MULTI_PARAM_LAYER_H
#define MULTI_PARAM_LAYER_H

#include "ilayer.h"
#include "iparam.h"
#include <vector>
#include <map>

template<MatMode mode, typename Dtype>
class IParamLayer : public ILayer<mode, Dtype>
{
public:
    IParamLayer(std::string _name, GraphAtt _at, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _at, _properr) 
    {
        this->graph_output = new GraphData<mode, Dtype>(DENSE);
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);	
    }
    
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override
    {
        ILayer<mode, Dtype>::UpdateOutput(prev_layer, sv, phase);
        
        auto& cur_output = GetImatState(this->graph_output, this->at)->DenseDerived();

		Dtype beta;				
		auto* param = this->GetParam(prev_layer->name);
        auto operand = this->GetOperand(prev_layer->name);
        
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
		} else
			beta = 1.0;
		
		param->InitializeBatch(prev_output, operand);
		param->UpdateOutput(GetImatState(prev_output, operand),  
                            &cur_output, beta, phase);
    }
    
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
    {
		auto* param = this->GetParam(prev_layer->name);
        auto operand = this->GetOperand(prev_layer->name);
        
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
		auto* param = this->GetParam(prev_layer->name);
        auto operand = this->GetOperand(prev_layer->name);
		
		auto& cur_grad = GetImatState(this->graph_gradoutput, this->at)->DenseDerived();
		
		param->AccDeriv(GetImatState(prev_layer->graph_output, operand), &cur_grad);
    }    

protected:
    virtual IParam<mode, Dtype>* GetParam(const std::string& prev_layer_name) = 0;
    virtual GraphAtt GetOperand(const std::string& prev_layer_name) = 0;
};

template<MatMode mode, typename Dtype>
class MultiParamLayer : public IParamLayer<mode, Dtype>
{
public:
	
	MultiParamLayer(std::string _name, GraphAtt _at, PropErr _properr = PropErr::T)
        : IParamLayer<mode, Dtype>(_name, _at, _properr) 
    {
		params_map.clear();
    }
	
	void AddParam(std::string prev_layer_name, IParam<mode, Dtype>* _param, GraphAtt _operand)
    {
        params_map[prev_layer_name] = std::make_pair(_param, _operand);
    }
	
	std::map< std::string, std::pair< IParam<mode, Dtype>*, GraphAtt > > params_map;	

protected:
    virtual IParam<mode, Dtype>* GetParam(const std::string& prev_layer_name) override
    {
        return this->params_map[prev_layer_name].first;   
    }
    
    virtual GraphAtt GetOperand(const std::string& prev_layer_name) override
    {
        return this->params_map[prev_layer_name].second;
    }
};

template<MatMode mode, typename Dtype>
class NodeLayer : public MultiParamLayer<mode, Dtype>
{
public:
	NodeLayer(std::string _name, PropErr _properr = PropErr::T) 
        : MultiParamLayer<mode, Dtype>(_name, GraphAtt::NODE, _properr) {}	
};

template<MatMode mode, typename Dtype>
class EdgeLayer : public MultiParamLayer<mode, Dtype>
{
public:
	EdgeLayer(std::string _name, PropErr _properr = PropErr::T) 
        : MultiParamLayer<mode, Dtype>(_name, GraphAtt::EDGE, _properr) {}	
};

template<MatMode mode, typename Dtype>
class SingleParamLayer : public IParamLayer<mode, Dtype>
{
public:
	
	SingleParamLayer(std::string _name, GraphAtt _at, IParam<mode, Dtype>* _param, GraphAtt _operand, PropErr _properr = PropErr::T)
        : IParamLayer<mode, Dtype>(_name, _at, _properr)
    {
        this->param = _param;
        this->operand = _operand;
    }
    
    IParam<mode, Dtype>* param;
    GraphAtt operand;

protected:
    virtual IParam<mode, Dtype>* GetParam(const std::string& prev_layer_name) override
    {
        return this->param;
    }
    
    virtual GraphAtt GetOperand(const std::string& prev_layer_name) override
    {
        return this->operand;
    }
};

template<MatMode mode, typename Dtype>
class SingleParamNodeLayer : public SingleParamLayer<mode, Dtype>
{
public:
	SingleParamNodeLayer(std::string _name, IParam<mode, Dtype>* _param, GraphAtt _operand, PropErr _properr = PropErr::T) 
        : SingleParamLayer<mode, Dtype>(_name, GraphAtt::NODE, _param, _operand, _properr) {}	
};

template<MatMode mode, typename Dtype>
class SingleParamEdgeLayer : public SingleParamLayer<mode, Dtype>
{
public:
	SingleParamEdgeLayer(std::string _name, IParam<mode, Dtype>* _param, GraphAtt _operand, PropErr _properr = PropErr::T) 
        : SingleParamLayer<mode, Dtype>(_name, GraphAtt::EDGE, _param, _operand, _properr) {}	
};

#endif