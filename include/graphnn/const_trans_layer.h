#ifndef CONST_TRANS_LAYER
#define CONST_TRANS_LAYER

#include "ilayer.h"

template<MatMode mode, typename Dtype>
class ConstTransLayer : public ILayer<mode, Dtype>
{
public:
    ConstTransLayer(std::string _name, GraphAtt _at, Dtype _a, Dtype _b, PropErr _properr = PropErr::T)
        : ILayer<mode, Dtype>(_name, _at, _properr), a(_a), b(_b)
        {
            this->graph_output = new GraphData<mode, Dtype>(DENSE);
            this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
        }
      
    virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override
    {
        auto& cur_output = GetImatState(this->graph_output, this->at)->DenseDerived();      
        auto* prev_output = prev_layer->graph_output;
        auto& prev_states = GetImatState(prev_output, this->at)->DenseDerived();
        Dtype beta;
        if (sv == SvType::WRITE2)
        {
            beta = 0.0;
            
            this->graph_output->graph = prev_output->graph;
            cur_output.Resize(prev_states.rows, prev_states.cols);
            
            if (this->at == GraphAtt::NODE)
                this->graph_output->edge_states = prev_output->edge_states; // edge state will remain the same
            else // EDGE
                this->graph_output->node_states = prev_output->node_states;
        } else 
            beta = 1.0;
                    
        cur_output.Axpby(a, prev_states, beta);
        cur_output.Add(b);
    }
    
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
    {
        auto& cur_grad = GetImatState(this->graph_gradoutput, this->at)->DenseDerived();
        auto& prev_grad = GetImatState(prev_layer->graph_gradoutput, operand)->DenseDerived(); 
        
        Dtype beta;
        if (sv == SvType::WRITE2)
        {
            beta = 0.0;
            prev_grad.Resize(cur_grad.rows, cur_grad.cols);
        } else
            beta = 1.0;
            
        prev_grad.Axpby(a, cur_grad, beta);
    }
};

#endif