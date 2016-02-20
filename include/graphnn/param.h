#ifndef PARAM_H
#define PARAM_H

#include "dense_matrix.h"
#include "sparse_matrix.h"

template<MatMode mode, typename Dtype>
class Param
{
    Param(std::string _name) : name(_name)
    {
        
    }
    
    DenseMat<mode, Dtype> value, grad;
    std::string name;
}; 

template<MatMode mode, typename Dtype>
class ConstSparseParam
{
    ConstSparseParam(std::string _name) : name(_name) {}
    
    SparseMat<mode, Dtype> value;
    std::string name;
};

template<MatMode mode, typename Dtype>
class IMessagePassParam : ConstSparseParam<mode, Dtype>
{
    IMessagePassParam(std::string _name);
    
protected:
    virtual void InitCPUWeight(GraphData<mode, Dtype>* g, GraphAtt operand) = 0;
    SparseMat<CPU, Dtype>* cpu_weight;           
};


#endif