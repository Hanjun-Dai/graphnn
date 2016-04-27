#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class ConcatLayer : public ILayer<mode, Dtype>
{
public:
    ConcatLayer(std::string _name, PropErr _properr = PropErr::T);        
    
    static std::string str_type()
    {
        return "Concat"; 
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override;
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;
    
protected:

    DenseMat<mode, Dtype> buf;    
};


#endif