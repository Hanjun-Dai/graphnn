#ifndef I_ACT_LAYER_H
#define I_ACT_LAYER_H

#include "i_layer.h"

enum class WriteType
{
	INPLACE = 0,
	OUTPLACE = 1	
};

template<MatMode mode, typename Dtype>
class IActLayer : public ILayer<mode, Dtype>
{
public:
	
	IActLayer(std::string _name, WriteType _wt, PropErr _properr = PropErr::T) 
        : ILayer<mode, Dtype>(_name, _properr), wt(_wt)
    {
        
    }
    
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
    {
        
    }
    
	WriteType wt;
};

#endif