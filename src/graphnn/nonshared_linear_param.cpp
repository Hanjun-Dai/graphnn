#include "nonshared_linear_param.h"
#include "dense_matrix.h"
#include "graph_data.h"
#include <cmath>

template<MatMode mode, typename Dtype>
NonsharedLinearParam<mode, Dtype>::NonsharedLinearParam(FILE* fid) 
										: LinearParam<mode, Dtype>(fid) {}										

template<MatMode mode, typename Dtype>
NonsharedLinearParam<mode, Dtype>::NonsharedLinearParam(std::string _name, size_t _input_size, size_t _output_size, BiasOption _bo)
							   			: LinearParam<mode, Dtype>(_name, _input_size, _output_size, _bo) {}

template<MatMode mode, typename Dtype>
NonsharedLinearParam<mode, Dtype>::NonsharedLinearParam(std::string _name, size_t _input_size, size_t _output_size, Dtype mean, Dtype std, BiasOption _bo)
										: LinearParam<mode, Dtype>(_name, _input_size, _output_size, mean, std, _bo) {}

template<MatMode mode, typename Dtype>		
void NonsharedLinearParam<mode, Dtype>::Reset(Dtype mean, Dtype std)
{
	this->weight.SetRandN(mean, std, this->input_size, this->output_size);
	if (this->bo == BiasOption::BIAS)
	{
		this->bias.Zeros(1, this->output_size);
	}
}

template<MatMode mode, typename Dtype>		
void NonsharedLinearParam<mode, Dtype>::InitializeBatch(GraphData<mode, Dtype>* g, GraphAtt operand)
{
	cur_input = cur_gradOutput = nullptr;
}

template<MatMode mode, typename Dtype>		
void NonsharedLinearParam<mode, Dtype>::AccDeriv(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* gradOutput)
{
	cur_input = input;
	cur_gradOutput = gradOutput;
}

template<MatMode mode, typename Dtype>		
void NonsharedLinearParam<mode, Dtype>::UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum)
{
	if (!cur_input || !cur_gradOutput)
		return;	
	if (momentum > 0)
	{
		throw "not implemented";
	} else 
	{							
		if (cur_input->GetMatType() == DENSE)		
			this->weight.GeMM(cur_input->DenseDerived(), *cur_gradOutput, Trans::T, Trans::N, -lr, (1 - l2_penalty * lr));
		else
			this->weight.SparseMM(cur_input->SparseDerived(), *cur_gradOutput, Trans::T, Trans::N, -lr, (1 - l2_penalty * lr));		
		
		if (this->bo == BiasOption::BIAS)
		{
			this->bias_multiplier.Resize(1, cur_input->rows);
			this->bias_multiplier.Fill(1.0);
			this->bias.GeMM(this->bias_multiplier, *cur_gradOutput, Trans::N, Trans::N, -lr, 1.0);			
		}
	}
}

template class NonsharedLinearParam<CPU, double>;
template class NonsharedLinearParam<CPU, float>;
template class NonsharedLinearParam<GPU, double>;
template class NonsharedLinearParam<GPU, float>;