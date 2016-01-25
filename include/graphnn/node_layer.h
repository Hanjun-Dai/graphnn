#ifndef NODE_LAYER_H
#define NODE_LAYER_H

#include "ilayer.h"
#include "iparam.h"
#include <vector>
#include <map>

template<MatMode mode, typename Dtype>
class NodeLayer : public ILayer<mode, Dtype>
{
public:
	typedef typename std::map<std::string, ILayer<mode, Dtype>* >::iterator layeriter;
	
	NodeLayer(std::string _name, PropErr _properr = PropErr::T);
	
	void AddParam(std::string prev_layer_name, IParam<mode, Dtype>* _param);	
	
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override;
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override;
	virtual void AccDeriv(ILayer<mode, Dtype>* prev_layer) override;
	
	std::map< std::string, std::vector< IParam<mode, Dtype>* > > params_map;	
	
private:
	void AddPrevLayerName(std::string _name);
};


#endif