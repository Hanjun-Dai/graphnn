#include "linear_param.h"
#include "dense_matrix.h"
#include "graph_data.h"
#include <cmath>

template<MatMode mode, typename Dtype>
LinearParam<mode, Dtype>::LinearParam(FILE* fid)
{
	this->Deserialize(fid);	
}

template<MatMode mode, typename Dtype>
LinearParam<mode, Dtype>::LinearParam(std::string _name, size_t _input_size, size_t _output_size, BiasOption _bo)
							   : IParam<mode, Dtype>(_name), bo(_bo)
{
	input_size = _input_size;
	output_size = _output_size;
	Reset(0, 1.0 / sqrt(output_size));						
}

template<MatMode mode, typename Dtype>
LinearParam<mode, Dtype>::LinearParam(std::string _name, size_t _input_size, size_t _output_size, Dtype mean, Dtype std, BiasOption _bo)
								: IParam<mode, Dtype>(_name), bo(_bo)
{
	input_size = _input_size;
	output_size = _output_size;
	Reset(mean, std);
}

template<MatMode mode, typename Dtype>		
void LinearParam<mode, Dtype>::Reset(Dtype mean, Dtype std)
{
	weight.SetRandN(mean, std, input_size, output_size);
	delta_weight.Zeros(input_size, output_size);
	acc_dweight.Zeros(input_size, output_size);
	if (bo == BiasOption::BIAS)
	{
		bias.Zeros(1, output_size);
		delta_bias.Zeros(1, output_size);
		acc_dbias.Zeros(1, output_size);
	}
}

template<MatMode mode, typename Dtype>		
void LinearParam<mode, Dtype>::InitializeBatch(GraphData<mode, Dtype>* g, GraphAtt operand)
{
	delta_weight.Zeros(input_size, output_size);
	if (bo == BiasOption::BIAS)
	{
		delta_bias.Zeros(1, output_size);
	}
}

template<MatMode mode, typename Dtype>		
void LinearParam<mode, Dtype>::UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase)
{
	if (input->GetMatType() == DENSE)
		output->GeMM(input->DenseDerived(), this->weight, Trans::N, Trans::N, 1.0, beta);
	else
		output->SparseMM(input->SparseDerived(), this->weight, Trans::N, Trans::N, 1.0, beta);
		
	if (bo == BiasOption::BIAS) 
	{
		output->AddRowVec(bias, 1.0);
	}
}

template<MatMode mode, typename Dtype>		
void LinearParam<mode, Dtype>::UpdateGradInput(IMatrix<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta)
{
    auto& prevGrad = gradInput->DenseDerived();
	prevGrad.GeMM(*gradOutput, this->weight, Trans::N, Trans::T, 1.0, beta);
}

template<MatMode mode, typename Dtype>		
void LinearParam<mode, Dtype>::AccDeriv(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* gradOutput)
{
	if (input->GetMatType() == DENSE)
		this->delta_weight.GeMM(input->DenseDerived(), *gradOutput, Trans::T, Trans::N, 1.0, 1.0);
	else
		this->delta_weight.SparseMM(input->SparseDerived(), *gradOutput, Trans::T, Trans::N, 1.0, 1.0);
							
	if (bo == BiasOption::BIAS)
	{
		bias_multiplier.Resize(1, input->rows);
		bias_multiplier.Fill(1.0);
		this->delta_bias.GeMM(bias_multiplier, *gradOutput, Trans::N, Trans::N, 1.0, 1.0);
	}
}

template<MatMode mode, typename Dtype>		
void LinearParam<mode, Dtype>::UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum)
{    
	if (momentum > 0)
	{
		delta_weight.Axpy(l2_penalty, weight);		
        acc_dweight.Axpby(lr, delta_weight, momentum);
		weight.Axpy(-1.0, acc_dweight);
		
		if (bo == BiasOption::BIAS)
		{
            acc_dbias.Axpby(lr, delta_bias, momentum);
			bias.Axpy(-1.0, acc_dbias);
		}
	} else 
	{	
        weight.Axpby(-lr, delta_weight, 1 - lr * l2_penalty);	
		
		if (bo == BiasOption::BIAS)
		{
			bias.Axpy(-lr, delta_bias);
		}
	}
}

template<MatMode mode, typename Dtype>		
void LinearParam<mode, Dtype>::Serialize(FILE* fid)
{
	IParam<mode, Dtype>::Serialize(fid);
	int i_bo = int(bo);
	assert(fwrite(&i_bo, sizeof(int), 1, fid) == 1);
	assert(fwrite(&input_size, sizeof(size_t), 1, fid) == 1);
	assert(fwrite(&output_size, sizeof(size_t), 1, fid) == 1);
	weight.Serialize(fid);
	delta_weight.Serialize(fid);
	acc_dweight.Serialize(fid);
	if (bo == BiasOption::BIAS)
	{
		bias.Serialize(fid);
		delta_bias.Serialize(fid);
		acc_dbias.Serialize(fid);
	}
}

template<MatMode mode, typename Dtype>		
void LinearParam<mode, Dtype>::Deserialize(FILE* fid)
{
	IParam<mode, Dtype>::Deserialize(fid);
	int i_bo;
	assert(fread(&i_bo, sizeof(int), 1, fid) == 1);
	bo = (BiasOption)i_bo;
	assert(fread(&input_size, sizeof(size_t), 1, fid) == 1);
	assert(fread(&output_size, sizeof(size_t), 1, fid) == 1);
	weight.Deserialize(fid);
	delta_weight.Deserialize(fid);
	acc_dweight.Deserialize(fid);
	if (bo == BiasOption::BIAS)
	{
		bias.Deserialize(fid);
		delta_bias.Deserialize(fid);
		acc_dbias.Deserialize(fid);
	}
}

template class LinearParam<CPU, double>;
template class LinearParam<CPU, float>;
template class LinearParam<GPU, double>;
template class LinearParam<GPU, float>;