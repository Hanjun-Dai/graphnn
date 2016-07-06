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

template<typename Dtype>
Dtype LossFunc<CPU, Dtype>::GetAverageRank(DenseMat<CPU, Dtype>& pred, SparseMat<CPU, Dtype>& label, RankOrder order)
{
        Dtype loss = 0.0;
        size_t offset = 0;
        for (size_t i = 0; i < pred.rows; ++i)
        {           
            unsigned cur_label = label.data->col_idx[i];
            Dtype cur_val = pred.data[offset + cur_label];
            for (size_t j = 0; j < pred.cols; ++j)
                if (j != cur_label)
                {
                    if (order == RankOrder::DESC && pred.data[offset + j] > cur_val)
                        loss++;
                    if (order == RankOrder::ASCE && pred.data[offset + j] < cur_val)
                        loss++;
                }                    
            offset += pred.cols;
        }
        loss += pred.rows;
        return loss;
}

template class LossFunc<CPU, float>;
template class LossFunc<CPU, double>;