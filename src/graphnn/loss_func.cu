#include "loss_func.h"
#include <cmath>
#include "cuda_helper.h"
#define min(x, y) (x < y ? x : y)
#define BLOCK_THREADS 128

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
Dtype LossFunc<GPU, Dtype>::GetLogLoss(DenseMat<GPU, Dtype>& pred, SparseMat<GPU, Dtype>& label)
{
        buf.Resize(label.data->nnz, 1);                 
        int thread_num = min(c_uCudaThreadNum, label.data->nnz);
        int blocksPerGrid = (label.data->nnz + thread_num - 1) / thread_num;
        LogLossKernel <<< blocksPerGrid, thread_num >>> (buf.data, pred.data, 
                                                         label.data->ptr, label.data->col_idx, label.data->val,
                                                         label.data->nnz, pred.rows, pred.cols); 
        Dtype loss = buf.Asum();
        return loss; 
}

template<typename Dtype>
__global__ void ErrCntKernel(Dtype *orig_ptr, Dtype* err_cnt, int* truth, int dim)
{
     __shared__ int buffer[BLOCK_THREADS + 1];
    Dtype* dst = orig_ptr + blockIdx.x * dim + blockIdx.y;
    
    int i_start = threadIdx.x;
    int i_end = dim;
    int i_step = blockDim.x;
    Dtype z;
    buffer[threadIdx.x] = i_start;
    Dtype cur_max = dst[i_start];
    for (int i = i_start; i < i_end; i += i_step)
    {
        z = dst[i];
        if (cur_max < z)
        {
            cur_max = z;
            buffer[threadIdx.x] = i;
        }
    }
    
    __syncthreads();

    // reduce
    if (threadIdx.x == 0)
    {
        int pred = buffer[0];
        int other;
        cur_max = dst[pred];
        for (int i = 1; i < blockDim.x; i++)
        {
            other = buffer[i];
            if(cur_max < dst[other])
            {
                cur_max = dst[other];
                pred = other;
            }
        }
        if (pred != truth[blockIdx.x])
            err_cnt[blockIdx.x] = 1.0;
        else
            err_cnt[blockIdx.x] = 0.0;
    }
}

template<typename Dtype>
Dtype LossFunc<GPU, Dtype>::GetErrCnt(DenseMat<GPU, Dtype>& pred, SparseMat<GPU, Dtype>& label)
{
        buf.Resize(label.data->nnz, 1);        
        dim3 blocks(pred.rows, 1);
        dim3 threads(min(BLOCK_THREADS, pred.cols));
        
        ErrCntKernel <<< blocks, threads, 0, GPUHandle::streams[pred.streamid] >>> (pred.data, buf.data, label.data->col_idx, pred.cols);  
        Dtype loss = buf.Asum();
        return loss; 
}

template<typename Dtype>
Dtype LossFunc<GPU, Dtype>::GetAverageRank(DenseMat<GPU, Dtype>& pred, SparseMat<GPU, Dtype>& label, RankOrder order)
{
        throw std::runtime_error("not implemented");
}

template class LossFunc<GPU, float>;
template class LossFunc<GPU, double>;