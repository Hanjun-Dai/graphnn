#ifndef SIMPLE_NODE_LAYER_H
#define SIMPLE_NODE_LAYER_H

#include "ilayer.h"
#include "iparam.h"

template<MatMode mode, typename Dtype>
class SimpleNodeLayer : public ILayer<mode, Dtype>
{
public:		
	SimpleNodeLayer(std::string _name, IParam<mode, Dtype>* _param, PropErr _properr = PropErr::T);
	    
    virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override;
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override;
	virtual void AccDeriv(ILayer<mode, Dtype>* prev_layer) override;
    
    
    IParam<mode, Dtype>* param;            
};

#endif