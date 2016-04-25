#ifndef GLOBAL_SUM_LAYER_H
#define GLOBAL_SUM_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class GlobalSumLayer : public ILayer<mode, Dtype>
{
public:
    GlobalSumLayer(std::string _name, PropErr _properr = PropErr::T);

    static std::string str_type()
    {
        return "GlobalPool"; 
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override;    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;    
    
protected:

    DenseMat<mode, Dtype> buf;    
};

#endif