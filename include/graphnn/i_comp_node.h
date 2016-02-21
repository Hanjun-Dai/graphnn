#ifndef I_COMP_NODE_H
#define I_COMP_NODE_H

#include "imatrix.h"

template<MatMode mode, typename Dtype>
class ICompNode
{
public:

    ICompNode() {}
        
    IMatrix<mode, Dtype>* value, *grad;
};

template<MatMode mode, typename Dtype>
class InputLayer : public ICompNode<mode, Dtype> 
{
public:

    InputLayer() : ICompNode<mode, Dtype>()
    {
        
    }
};

template<MatMode mode, typename Dtype>
class SumLayer : public ICompNode<mode, Dtype>
{
public:
    
    SumLayer() : ICompNode<mode, Dtype>()
    {
        this->value = new DenseMat<mode, Dtype>();
        this->grad = new DenseMat<mode, Dtype>();
    }
};


#endif