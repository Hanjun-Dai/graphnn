#ifndef ILAYER_H
#define ILAYER_H

#include "imatrix.h"
#include <vector>

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
    ILayer(std::string _name, PropErr _properr = PropErr::T) 
            : name(_name), properr(_properr)
            {
            }
            
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) = 0;     
                    
    std::string name;
    PropErr properr;
    IMatrix<mode, Dtype>* state, *grad;
};

#endif