#ifndef I_ACT_LAYER_H
#define I_ACT_LAYER_H

#include "ilayer.h"

enum class WriteType
{
	INPLACE = 0,
	OUTPLACE = 1	
};

enum class ActTarget
{
	NODE = 0,
	EDGE = 1,
	NODE_EDGE = 2	
};

template<MatMode mode, typename Dtype>
class IActLayer : public ILayer<mode, Dtype>
{
public:
	
	IActLayer(std::string _name, WriteType _wt, ActTarget _at, PropErr _properr = PropErr::T) : ILayer<mode, Dtype>(_name, _properr), wt(_wt), at(_at) 
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
		
		if (this->at == ActTarget::NODE || this->at == ActTarget::NODE_EDGE)
			Act(prev_output->node_states->DenseDerived(), this->graph_output->node_states->DenseDerived());
		if (this->at == ActTarget::EDGE || this->at == ActTarget::NODE_EDGE)
			Act(prev_output->edge_states->DenseDerived(), this->graph_output->edge_states->DenseDerived());        
    }

    virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override
    {
		if (this->at == ActTarget::NODE || this->at == ActTarget::NODE_EDGE)
		{
			auto& prev_grad = prev_layer->graph_gradoutput->node_states->DenseDerived();			
			auto& cur_grad = this->graph_gradoutput->node_states->DenseDerived();            
            auto& prev_output = prev_layer->graph_output->node_states->DenseDerived();
			auto& cur_output = this->graph_output->node_states->DenseDerived();
            
            Derivative(prev_grad, prev_output, cur_output, cur_grad);
		}
		
		if (this->at == ActTarget::EDGE || this->at == ActTarget::NODE_EDGE)
		{
			throw "not implemented";
		}
    }
    


    virtual void Act(DenseMat<mode, Dtype>& prev_out, DenseMat<mode, Dtype>& cur_out) = 0;
    virtual void Derivative(DenseMat<mode, Dtype>& dst, DenseMat<mode, Dtype>& prev_output, DenseMat<mode, Dtype>& cur_output, DenseMat<mode, Dtype>& cur_grad) = 0;
	
	WriteType wt;
	ActTarget at;
};

/*
template<MatMode mode, typename Dtype>
class SigmoidLayer : public IActLayer<mode, Dtype>
{
public:
	typedef typename std::map<std::string, ILayer<mode, Dtype>* >::iterator layeriter;		
	SigmoidLayer(std::string _name, WriteType _wt, ActTarget _at, PropErr _properr = PropErr::T);
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override;
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override;
};

template<MatMode mode, typename Dtype>
class SoftmaxLayer : public IActLayer<mode, Dtype>
{
public:
	typedef typename std::map<std::string, ILayer<mode, Dtype>* >::iterator layeriter;
	SoftmaxLayer(std::string _name, WriteType _wt, ActTarget _at, PropErr _properr = PropErr::T);
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override;
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override;
};
*/
#endif