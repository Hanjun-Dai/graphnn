#include "batch_norm_param.h"
#include "dense_matrix.h"
#include "graph_data.h"
#include <cmath>
#include <algorithm>

template<MatMode mode, typename Dtype>
BatchNormParam<mode, Dtype>::BatchNormParam(FILE* fid)
{
	this->Deserialize(fid);
}

template<MatMode mode, typename Dtype>
BatchNormParam<mode, Dtype>::BatchNormParam(std::string _name, size_t _input_size, bool _parametrized, Dtype _eps, Dtype _smooth)
									: IParam<mode, Dtype>(_name), input_size(_input_size), eps(_eps), smooth(_smooth)
{
    parametrized = _parametrized; 
    acc_mean.Zeros(1, input_size);
    acc_inv_std.Resize(1, input_size);
    acc_inv_std.Fill(1.0);
    
    if (parametrized)
    {
        scale.Resize(1, input_size);
        scale.SetRandN(0, 0.1);
        bias.Zeros(1, input_size);
    }
    cur_grad_output = nullptr;
}

template<MatMode mode, typename Dtype>		
void BatchNormParam<mode, Dtype>::UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase)
{
    assert(beta == 0);
    output->CopyFrom(input->DenseDerived());
    if (phase == TEST)
    {        
        output->AddRowVec(acc_mean, -1.0);
        output->MulRowVec(acc_inv_std);
    } else {
        cur_inv_std.Mean(input->DenseDerived()); 
        acc_mean.Axpby(smooth, cur_inv_std, 1 - smooth);
        
        output->AddRowVec(cur_inv_std, -1.0);
        
        mat_buffer.CopyFrom(output);
        mat_buffer.Square();
        
        cur_inv_std.Mean(mat_buffer);
        cur_inv_std.Add(eps);
        cur_inv_std.InvSqrt();
        
        acc_inv_std.Axpby(smooth, cur_inv_std, 1 - smooth);
        
        output->MulRowVec(cur_inv_std);
    }    
    normed_output.CopyFrom(*output);
    
    if (parametrized)
    {
        output->MulRowVec(scale); 
        output->AddRowVec(bias, 1.0);
    }
}

// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
//
// dE(Y)/dX =
//   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
//     ./ sqrt(var(X) + eps)
//
template<MatMode mode, typename Dtype>		
void BatchNormParam<mode, Dtype>::UpdateGradInput(DenseMat<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta)
{
    // dE/dY \cdot Y
    mat_buffer.EleWiseMul(*gradOutput, normed_output);
    // mean(dE/dY \cdot Y)
    row_buffer.Mean(mat_buffer);
    // mean(dE/dY \cdot Y) \cdot Y
    mat_buffer.MulRowVec(normed_output, row_buffer);
    
    // mean(dE/dY)
    row_buffer.Mean(*gradOutput);
    // mean(dE/dY) + mean(dE/dY \cdot Y) \cdot Y
    mat_buffer.AddRowVec(row_buffer, 1.0);
    
    // dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y
    mat_buffer.Axpby(1.0, *gradOutput, -1.0);
    
    // dE(Y)/dX
    mat_buffer.MulRowVec(cur_inv_std);
    
    if (parametrized)
        mat_buffer.MulRowVec(scale);
        
    gradInput->Axpby(1.0, mat_buffer, beta);                    
}

template<MatMode mode, typename Dtype>		
void BatchNormParam<mode, Dtype>::AccDeriv(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* gradOutput)
{
    cur_grad_output = gradOutput;
}

template<MatMode mode, typename Dtype>		
void BatchNormParam<mode, Dtype>::UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum)
{
    if (parametrized)
    {
        row_multiplier.Resize(1, cur_grad_output->rows);
        row_multiplier.Fill(1.0);
        mat_buffer.EleWiseMul(*cur_grad_output, normed_output); 
        scale.GeMM(row_multiplier, mat_buffer, Trans::N, Trans::N, -lr, (1 - l2_penalty * lr));        	
        bias.GeMM(row_multiplier, *cur_grad_output, Trans::N, Trans::N, -lr, 1.0);
    }
}

template<MatMode mode, typename Dtype>		
void BatchNormParam<mode, Dtype>::Serialize(FILE* fid)
{
	IParam<mode, Dtype>::Serialize(fid);
    assert(fwrite(&input_size, sizeof(size_t), 1, fid) == 1);
    assert(fwrite(&eps, sizeof(Dtype), 1, fid) == 1);
    assert(fwrite(&smooth, sizeof(Dtype), 1, fid) == 1);
    
    acc_mean.Serialize(fid);
    acc_inv_std.Serialize(fid);
    scale.Serialize(fid);
    bias.Serialize(fid);        
}

template<MatMode mode, typename Dtype>		
void BatchNormParam<mode, Dtype>::Deserialize(FILE* fid)
{    
	IParam<mode, Dtype>::Deserialize(fid);
    assert(fread(&input_size, sizeof(size_t), 1, fid) == 1);
    assert(fread(&eps, sizeof(Dtype), 1, fid) == 1);
    assert(fread(&smooth, sizeof(Dtype), 1, fid) == 1);
    
    acc_mean.Deserialize(fid);
    acc_inv_std.Deserialize(fid);
    scale.Deserialize(fid);
    bias.Deserialize(fid);
}

template class BatchNormParam<CPU, double>;
template class BatchNormParam<CPU, float>;

template class BatchNormParam<GPU, double>;
template class BatchNormParam<GPU, float>;
