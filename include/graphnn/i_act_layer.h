#ifndef I_ACT_LAYER_H
#define I_ACT_LAYER_H

#include "ilayer.h"

enum class WriteType
{
	INPLACE = 0,
	OUTPLACE = 1	
};

template<MatMode mode, typename Dtype>
class IActLayer : public ILayer<mode, Dtype>
{
public:
	
	IActLayer(std::string _name, GraphAtt _at, WriteType _wt, PropErr _properr = PropErr::T) : ILayer<mode, Dtype>(_name, _at, _properr), wt(_wt)
    {
        this->graph_output = new GraphData<mode, Dtype>(DENSE);		
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
    }

    virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override
    {
        assert(sv == SvType::WRITE2);
        auto* prev_output = prev_layer->graph_output;
		this->graph_output->graph = prev_output->graph;
		
		if (this->wt == WriteType::INPLACE)
		{
			this->graph_output->node_states = prev_output->node_states;
			this->graph_output->edge_states = prev_output->edge_states; 
		} else 
		{
			this->graph_output->node_states->DenseDerived().Resize(prev_output->node_states->rows, prev_output->node_states->cols);
			this->graph_output->edge_states->DenseDerived().Resize(prev_output->edge_states->rows, prev_output->edge_states->cols);
		}
		
        auto& prev_state = GetImatState(prev_output, this->at)->DenseDerived();
        auto& cur_state = GetImatState(this->graph_output, this->at)->DenseDerived();
         
        Act(prev_state, cur_state);        
    }

    virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
    {
        auto& prev_grad = GetImatState(prev_layer->graph_gradoutput, this->at)->DenseDerived();			
		auto& cur_grad = GetImatState(this->graph_gradoutput, this->at)->DenseDerived();            
        auto& prev_output = GetImatState(prev_layer->graph_output, this->at)->DenseDerived();
		auto& cur_output = GetImatState(this->graph_output, this->at)->DenseDerived();
                        
        prev_grad.Resize(cur_grad.rows, cur_grad.cols);
        Derivative(prev_grad, prev_output, cur_output, cur_grad);        
    }
    
    virtual void Act(DenseMat<mode, Dtype>& prev_out, DenseMat<mode, Dtype>& cur_out) = 0;
    virtual void Derivative(DenseMat<mode, Dtype>& dst, DenseMat<mode, Dtype>& prev_output, DenseMat<mode, Dtype>& cur_output, DenseMat<mode, Dtype>& cur_grad) = 0;
	
	WriteType wt;
};

/*
template<MatMode mode, typename Dtype>
class SigmoidLayer : public IActLayer<mode, Dtype>
{
public:
	typedef typename std::map<std::string, ILayer<mode, Dtype>* >::iterator layeriter;		
	SigmoidLayer(std::string _name, WriteType _wt, GraphAtt _at, PropErr _properr = PropErr::T);
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override;
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override;
};

template<MatMode mode, typename Dtype>
class SoftmaxLayer : public IActLayer<mode, Dtype>
{
public:
	typedef typename std::map<std::string, ILayer<mode, Dtype>* >::iterator layeriter;
	SoftmaxLayer(std::string _name, WriteType _wt, GraphAtt _at, PropErr _properr = PropErr::T);
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override;
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override;
};
*/
#endif