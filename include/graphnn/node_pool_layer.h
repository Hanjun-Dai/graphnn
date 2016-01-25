#ifndef NODE_POOL_LAYER_H
#define NODE_POOL_LAYER_H

#include "ilayer.h"
#include "node_pool_param.h"

template<MatMode mode, typename Dtype>
class NodePoolLayer : public ILayer<mode, Dtype>
{
public:
	typedef typename std::map<std::string, ILayer<mode, Dtype>* >::iterator layeriter;
	
	NodePoolLayer(std::string _name, NodePoolParam<mode, Dtype>* _param, PropErr _properr = PropErr::T);
	
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override;
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override;
	
	NodePoolParam<mode, Dtype>* param;
};

#endif