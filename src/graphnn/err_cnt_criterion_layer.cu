#include "err_cnt_criterion_layer.h"
#include <cmath>
#include "cuda_helper.h"
#define min(x, y) (x < y ? x : y)
#define BLOCK_THREADS 128

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
Dtype GetErrCnt(DenseMat<GPU, Dtype>& pred, SparseMat<GPU, Dtype>& label, DenseMat<GPU, Dtype>& buf)
{        
        dim3 blocks(pred.rows, 1);
        dim3 threads(min(BLOCK_THREADS, pred.cols));
        
        ErrCntKernel <<< blocks, threads, 0, GPUHandle::streams[pred.streamid] >>> (pred.data, buf.data, label.data->col_idx, pred.cols);  
        Dtype loss = buf.Asum();
        return loss; 
}

template<typename Dtype>
Dtype GetErrCnt(DenseMat<CPU, Dtype>& pred, SparseMat<CPU, Dtype>& label, DenseMat<CPU, Dtype>& buf)
{
        assert(pred.rows == buf.rows);
        Dtype loss = 0.0;
        for (size_t i = 0; i < pred.rows; ++i)
        {
            if (pred.GetRowMaxIdx(i) != (unsigned)label.data->col_idx[i])
                loss++;
        }
        return loss;
}

template<MatMode mode, typename Dtype>
ErrCntCriterionLayer<mode, Dtype>::ErrCntCriterionLayer(std::string _name)
								 : ICriterionLayer<mode, Dtype>(_name, PropErr::N)
{
		this->graph_gradoutput = nullptr;
        this->graph_output = nullptr;
}

template<MatMode mode, typename Dtype>
Dtype ErrCntCriterionLayer<mode, Dtype>::GetLoss(GraphData<mode, Dtype>* graph_truth)
{                                
        auto& pred = this->graph_output->node_states->DenseDerived();
        auto& labels = graph_truth->node_states->SparseDerived(); 
        buf.Resize(labels.data->nnz, 1);                      
        
		Dtype loss = GetErrCnt(pred, labels, buf);
        return loss;
}

template class ErrCntCriterionLayer<CPU, float>;
template class ErrCntCriterionLayer<CPU, double>;
template class ErrCntCriterionLayer<GPU, float>;
template class ErrCntCriterionLayer<GPU, double>;