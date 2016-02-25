#include "classnll_criterion_layer.h"
#include <cmath>
#include "cuda_helper.h"

template<typename Dtype>
__global__ void LogLossKernel(Dtype* dst, Dtype* pred, 
                              int* row_ptr, int* col_idx, Dtype* val, 
                              int nnz, int n_rows, int n_cols)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < nnz)
    {
        int row = get_sp_row_idx(i, row_ptr, n_rows);
        dst[i] = cuda_log(pred[row * n_cols + col_idx[i]]) * val[i];
    }
}

template<typename Dtype>
Dtype GetLogLoss(DenseMat<GPU, Dtype>& pred, SparseMat<GPU, Dtype>& label, DenseMat<GPU, Dtype>& buf)
{        
        int thread_num = min(c_uCudaThreadNum, label.data->nnz);
        int blocksPerGrid = (label.data->nnz + thread_num - 1) / thread_num;
        LogLossKernel <<< blocksPerGrid, thread_num >>> (buf.data, pred.data, 
                                                         label.data->ptr, label.data->col_idx, label.data->val,
                                                         label.data->nnz, pred.rows, pred.cols); 
        Dtype loss = buf.Asum();
        return loss; 
}

template<typename Dtype>
Dtype GetLogLoss(DenseMat<CPU, Dtype>& pred, SparseMat<CPU, Dtype>& label, DenseMat<CPU, Dtype>& buf)
{
        assert(pred.rows == buf.rows);
        Dtype loss = 0.0;
        for (size_t i = 0; i < label.rows; ++i)
        {
            for (int k = label.data->ptr[i]; k < label.data->ptr[i + 1]; ++k)
                loss -= log(pred.data[label.cols * i + label.data->col_idx[k]]) * label.data->val[k];
        }
        return loss;
} 

template class ClassNLLCriterionLayer<CPU, float>;
template class ClassNLLCriterionLayer<CPU, double>;
template class ClassNLLCriterionLayer<GPU, float>;
template class ClassNLLCriterionLayer<GPU, double>;