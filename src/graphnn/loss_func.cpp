#include "loss_func.h"
#include <cmath>

template<typename Dtype>
Dtype LossFunc<CPU, Dtype>::GetLogLoss(DenseMat<CPU, Dtype>& pred, SparseMat<CPU, Dtype>& label)
{
        Dtype loss = 0.0;
        for (size_t i = 0; i < label.rows; ++i)
        {
            for (int k = label.data->ptr[i]; k < label.data->ptr[i + 1]; ++k)
                loss -= log(pred.data[label.cols * i + label.data->col_idx[k]]) * label.data->val[k];
        }
        return loss;
} 

template<typename Dtype>
Dtype LossFunc<CPU, Dtype>::GetErrCnt(DenseMat<CPU, Dtype>& pred, SparseMat<CPU, Dtype>& label)
{
        Dtype loss = 0.0;
        for (size_t i = 0; i < pred.rows; ++i)
        {
            if (pred.GetRowMaxIdx(i) != (unsigned)label.data->col_idx[i])
                loss++;
        }
        return loss;
}

template class LossFunc<CPU, float>;
template class LossFunc<CPU, double>;