#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "ilayer.h"

template<MatMode mode, typename Dtype>
class InputLayer : public ILayer<mode, Dtype>
{
public:
	InputLayer(std::string _name, PropErr _properr = PropErr::N)
            : ILayer<mode, Dtype>(_name, _properr) {}
	virtual void UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase) override {}
	virtual void BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv) override {}
};

#endif