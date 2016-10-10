#include "relu_layer.h"
#include "sigmoid_layer.h"
#include "tanh_layer.h"
#include "softmax_layer.h"
#include "multinomial_sample_layer.h"
#include "mkl_helper.h"
#include "dense_matrix.h"

// =========================================== relu layer ================================================
template<typename Dtype>
void ReLULayer<CPU, Dtype>::Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out)
{
        for (size_t i = 0; i < cur_out.count; ++i)
            cur_out.data[i] = prev_out.data[i] > 0 ? prev_out.data[i] : 0; 
}

template<typename Dtype>
void ReLULayer<CPU, Dtype>::Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                            DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad, Dtype beta)
{
        dst.Scale(beta);
        for (int i = 0; i < dst.count; ++i)
            if (cur_output.data[i] > 0.0)
                dst.data[i] += cur_grad.data[i];
}

template class ReLULayer<CPU, float>;
template class ReLULayer<CPU, double>;

// =========================================== sigmoid layer ================================================

template<typename Dtype>
void SigmoidLayer<CPU, Dtype>::Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out)
{
    for (size_t i = 0; i < cur_out.count; ++i)
        cur_out.data[i] = 1.0 / (1.0 + exp(-prev_out.data[i])); 
}

template<typename Dtype>
void SigmoidLayer<CPU, Dtype>::Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                            DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad, Dtype beta)
{
    dst.Scale(beta);
    for (size_t i = 0; i < cur_output.count; ++i)
		dst.data[i] += cur_grad.data[i] * cur_output.data[i] * (1 - cur_output.data[i]);        	
}

template class SigmoidLayer<CPU, float>;
template class SigmoidLayer<CPU, double>;

// =========================================== tanh layer ================================================

template<typename Dtype>
void TanhLayer<CPU, Dtype>::Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out)
{
    Dtype x, y;
    for (size_t i = 0; i < cur_out.count; ++i)
    {
        x = exp(prev_out.data[i]); 
        y = exp(-prev_out.data[i]);
        cur_out.data[i] = (x - y) / (x + y);
    }
}

template<typename Dtype>
void TanhLayer<CPU, Dtype>::Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                            DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad, Dtype beta)
{
    dst.Scale(beta);
    for (size_t i = 0; i < cur_output.count; ++i)
		dst.data[i] += cur_grad.data[i] * (1 - cur_output.data[i] * cur_output.data[i]);        	
}

template class TanhLayer<CPU, float>;
template class TanhLayer<CPU, double>;

// =========================================== softmax layer ================================================

template<typename Dtype>
void SoftmaxLayer<CPU, Dtype>::Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out)
{
        if (&cur_out != &prev_out)
            cur_out.CopyFrom(prev_out);
        cur_out.Softmax();
}

template<typename Dtype>
void SoftmaxLayer<CPU, Dtype>::Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                               DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad, Dtype beta)
{
    buf.CopyFrom(cur_grad);
    
    Dtype z;    
    size_t offset = 0;
    for (size_t i = 0; i < buf.rows; ++i)
    {
        z = MKLHelper_Dot(buf.cols, cur_grad.data + offset, cur_output.data + offset);
        
        for (size_t j = 0; j < buf.cols; ++j)
            buf.data[offset + j] -= z;
        
        offset += buf.cols; 
    }
    
    buf.EleWiseMul(cur_output);
    
    dst.Axpby(1.0, buf, beta);
}

template class SoftmaxLayer<CPU, float>;
template class SoftmaxLayer<CPU, double>;

// =========================================== multinomial_sample layer ================================================

template<typename Dtype>
void MultinomialSampleLayer<CPU, Dtype>::Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out)
{
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
    std::uniform_real_distribution<Dtype> distribution(0, 1);

    cur_out.Zeros(prev_out.rows, prev_out.cols);
    Dtype* prob = prev_out.data;        
    for (size_t i = 0; i < prev_out.rows; ++i)
    {
        size_t max_id = 0;
        Dtype cum_sum = 0, threshold = distribution(generator);
        for (size_t j = 0; j < prev_out.cols; ++j)
        {
            if (st == SampleType::MAX && prob[j] > prob[max_id])
                max_id = j;
            if (st == SampleType::STOCHASTIC)
            {
                cum_sum += prob[j];
                if (cum_sum >= threshold)
                {
                    max_id = j;
                    break;
                }
            }
        }
        cur_out.data[cur_out.cols * i + max_id] = 1.0;
        prob += prev_out.cols;
    }
}

template<typename Dtype>
void MultinomialSampleLayer<CPU, Dtype>::Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                               DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad, Dtype beta)
{
    dst.Axpby(1.0, cur_grad, beta);
}

template class MultinomialSampleLayer<CPU, float>;
template class MultinomialSampleLayer<CPU, double>;