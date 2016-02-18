#ifndef ILAYER_H
#define ILAYER_H

#include "imatrix.h"
#include "graph_data.h"
#include <string>
#include <chrono>

enum class PropErr
{
	N = 0,
	T = 1	
};

enum class SvType
{
	WRITE2 = 0,
	ADD2 = 1	
};

template<MatMode mode, typename Dtype>
class ILayer
{
public:
	typedef typename std::map<std::string, ILayer<mode, Dtype>* >::iterator layeriter;
	
	ILayer(std::string _name, GraphAtt _at, PropErr _properr = PropErr::T) 
            : name(_name), properr(_properr), at(_at) 
	{
		graph_output = nullptr;
		graph_gradoutput = nullptr;
	}
	
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase)
    {
        auto* prev_output = prev_layer->graph_output;
        if (sv == SvType::WRITE2)
		{
            // we assume all the prev layers have the same graph structure;
			this->graph_output->graph = prev_output->graph;

            if (this->at == GraphAtt::NODE)
                this->graph_output->edge_states = prev_output->edge_states; // edge state will remain the same
            else // EDGE
                this->graph_output->node_states = prev_output->node_states;
		}
    }
    
    virtual void Clear() {}
    
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) = 0;
	virtual void AccDeriv(ILayer<mode, Dtype>* prev_layer) { }
	
	std::string name;
	
	GraphData<mode, Dtype> *graph_output, *graph_gradoutput;
	PropErr properr;
    GraphAtt at;
};


#endif