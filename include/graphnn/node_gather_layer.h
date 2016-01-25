#ifndef NODE_GATHER_LAYER_H
#define NODE_GATHER_LAYER_H

#include "ilayer.h"
#include "iparam.h"

template<MatMode mode, typename Dtype>
class NodeGatherLayer : public ILayer<mode, Dtype>
{
public:
	typedef typename std::map<std::string, ILayer<mode, Dtype>* >::iterator layeriter;
	
	NodeGatherLayer(std::string _name, PoolingOp _op, PropErr _properr = PropErr::T);
	
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override;
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override;
	
	PoolingOp op;
};

#endif