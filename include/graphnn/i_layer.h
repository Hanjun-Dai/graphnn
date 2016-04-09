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
                this->state = this->grad = nullptr;
            }
            
    virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) = 0;    
    
    virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) = 0;
    
    virtual bool HasParam()
    {
        return false;
    }
    
    virtual bool IsSupervised()
    {
        return false;
    }                
                    
    std::string name;
    PropErr properr;
    IMatrix<mode, Dtype>* state, *grad;
};

template<MatMode mode, typename Dtype>
class IParametric
{
public:
        virtual void AccDeriv(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx) = 0;
};

#endif