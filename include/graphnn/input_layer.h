#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class InputLayer : public ILayer<mode, Dtype>
{
public:
	InputLayer(std::string _name)
            : ILayer<mode, Dtype>(_name, PropErr::N) {}

    static std::string str_type()
    {
        return "Input"; 
    }

    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override {}
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override {}
};

#endif