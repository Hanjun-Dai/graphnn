#include "relu_layer.h"
#include "sin_layer.h"
#include "cos_layer.h"
#include "exp_layer.h"
#include "softmax_layer.h"
#include "dense_matrix.h"
#include "sigmoid_layer.h"
#include "mkl_helper.h"

// =========================================== relu layer ================================================
template<typename Dtype>
void ReLULayer<CPU, Dtype>::Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out)
{
        for (size_t i = 0; i < cur_out.count; ++i)
            cur_out.data[i] = prev_out.data[i] > 0 ? prev_out.data[i] : 0; 
}

template<typename Dtype>
void ReLULayer<CPU, Dtype>::Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                            DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad)
{
        dst.CopyFrom(cur_grad);
        for (int i = 0; i < dst.count; ++i)
            if (cur_output.data[i] <= 0.0)
                dst.data[i] = 0.0;
}

template class ReLULayer<CPU, float>;
template class ReLULayer<CPU, double>;

// =========================================== sin layer ================================================

template class SinLayer<CPU, float>;
template class SinLayer<CPU, double>;

// =========================================== cos layer ================================================

template class CosLayer<CPU, float>;
template class CosLayer<CPU, double>;

// =========================================== exp layer ================================================

template class ExpLayer<CPU, float>;
template class ExpLayer<CPU, double>;

// =========================================== softmax layer ================================================

template class SoftmaxLayer<CPU, float>;
template class SoftmaxLayer<CPU, double>;


// =========================================== sigmoid layer ================================================

template<typename Dtype>
void SigmoidLayer<CPU, Dtype>::Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out)
{
    for (size_t i = 0; i < cur_out.count; ++i)
        cur_out.data[i] = 1.0 / (1.0 + exp(-prev_out.data[i])); 
}

template<typename Dtype>
void SigmoidLayer<CPU, Dtype>::Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                            DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad)
{
    for (size_t i = 0; i < cur_output.count; ++i)
		dst.data[i] = cur_grad.data[i] * cur_output.data[i] * (1 - cur_output.data[i]);        	
}

template class SigmoidLayer<CPU, float>;
template class SigmoidLayer<CPU, double>;