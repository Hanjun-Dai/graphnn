#ifndef ILAYER_H
#define ILAYER_H

#include "imatrix.h"

enum class PropErr
{
	N = 0,
	T = 1	
};

template<MatMode mode, typename Dtype>
class NNGraph;

template<MatMode mode, typename Dtype>
class ILayer
{
public:
    ILayer(NNGraph<mode, Dtype>* _nn, PropErr _properr = PropErr::T) 
            : properr(_properr)
            {
                this->nn = _nn;  
            }
            
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands) = 0;
            
    IMatrix<mode, Dtype>* state, *grad;
    
    NNGraph<mode, Dtype>* nn;
    PropErr properr;
};

#endif