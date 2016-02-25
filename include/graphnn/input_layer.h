#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class InputLayer : public ILayer<mode, Dtype>
{
public:
	InputLayer(std::string _name)
            : ILayer<mode, Dtype>(_name, PropErr::N) {}

};

#endif