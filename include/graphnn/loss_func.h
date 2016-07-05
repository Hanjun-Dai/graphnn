#ifndef LOSS_FUNC_H
#define LOSS_FUNC_H

#include "dense_matrix.h"
#include "sparse_matrix.h"

template<MatMode mode, typename Dtype>
class LossFunc;

template<typename Dtype>
class LossFunc<CPU, Dtype>
{
public:    
    static Dtype GetLogLoss(DenseMat<CPU, Dtype>& pred, SparseMat<CPU, Dtype>& label);
    static Dtype GetErrCnt(DenseMat<CPU, Dtype>& pred, SparseMat<CPU, Dtype>& label);
    static Dtype GetAverageRank(DenseMat<CPU, Dtype>& pred, SparseMat<CPU, Dtype>& label);
};

template<typename Dtype>
class LossFunc<GPU, Dtype>
{
public:
    static Dtype GetLogLoss(DenseMat<GPU, Dtype>& pred, SparseMat<GPU, Dtype>& label);
    static Dtype GetErrCnt(DenseMat<GPU, Dtype>& pred, SparseMat<GPU, Dtype>& label);
    static Dtype GetAverageRank(DenseMat<GPU, Dtype>& pred, SparseMat<GPU, Dtype>& label);

private:    
    static DenseMat<GPU, Dtype> buf;    
};

template<typename Dtype>
DenseMat<GPU, Dtype> LossFunc<GPU, Dtype>::buf;




#endif