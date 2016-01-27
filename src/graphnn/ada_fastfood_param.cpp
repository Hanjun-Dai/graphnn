#include "ada_fastfood_param.h"
#include "dense_matrix.h"
#include "graph_data.h"
#include "fast_wht.h"
#include <cmath>
#include <algorithm>

template<MatMode mode, typename Dtype>
AdaFastfoodParam<mode, Dtype>::AdaFastfoodParam(FILE* fid)
{
	this->Deserialize(fid);
}

template<MatMode mode, typename Dtype>
AdaFastfoodParam<mode, Dtype>::AdaFastfoodParam(std::string _name, GraphAtt _operand, size_t _input_size, size_t _output_size, BiasOption _bo)
									: AdaFastfoodParam<mode, Dtype>(_name, _operand, _input_size, _output_size, 0.0, 1.0 / sqrt(output_size), _bo) {}

template<MatMode mode, typename Dtype>
AdaFastfoodParam<mode, Dtype>::AdaFastfoodParam(std::string _name, GraphAtt _operand, size_t _input_size, size_t _output_size, Dtype mean, Dtype std, BiasOption _bo)
									: IParam<mode, Dtype>(_name, _operand), bo(_bo)
{
	input_size = _input_size;
	output_size = _output_size;
	perm.clear();
    inv_perm.clear();		
	binomial.clear();
	gaussian.clear();
	chisquare.clear();
	state_B.clear();
    state_G.clear();
	state_P.clear();
	state_S.clear();
	grad_buf.clear();
        
	pad_input_size = (int)pow(2, ceil(log2(input_size)));
	pad_output_size = (int)pow(2, ceil(log2(output_size)));	
		
	if (pad_output_size < pad_input_size)
		num_parts = 1;
	else
		num_parts = pad_output_size / pad_input_size; 
	grad_updated = false;
	InitStructures();
	InitParams(mean, std);	
}

template<MatMode mode, typename Dtype>
void AdaFastfoodParam<mode, Dtype>::InitStructures()
{
	wht_helper = new FastWHT<mode, Dtype>((unsigned int)log2(pad_input_size));
	for (size_t p = 0; p < num_parts; ++p)
	{
        int* perm_ptr, *inv_perm_ptr;
        MatUtils<mode>::MallocArr(perm_ptr, sizeof(int) * pad_input_size);
        MatUtils<mode>::MallocArr(inv_perm_ptr, sizeof(int) * pad_input_size);
		perm.push_back(perm_ptr);
		inv_perm.push_back(inv_perm_ptr);
		
		binomial.push_back(new DenseMat<mode, Dtype>(1, pad_input_size));
		gaussian.push_back(new DenseMat<mode, Dtype>(1, pad_input_size));
		chisquare.push_back(new DenseMat<mode, Dtype>(1, pad_input_size));
        
		state_B.push_back(new DenseMat<mode, Dtype>());
		state_G.push_back(new DenseMat<mode, Dtype>());
		state_P.push_back(new DenseMat<mode, Dtype>());
		state_S.push_back(new DenseMat<mode, Dtype>());
		grad_buf.push_back(new DenseMat<mode, Dtype>());
	}
}

template<MatMode mode, typename Dtype>
void AdaFastfoodParam<mode, Dtype>::InitParams(Dtype mean, Dtype std)
{
	bias.Zeros(1, pad_output_size);
    int* perm_buf = new int[pad_input_size];
    int* inv_perm_buf = new int[pad_input_size];
    
	for (size_t p = 0; p < num_parts; ++p)
	{
		for (size_t i = 0; i < pad_input_size; ++i)
			perm_buf[i] = i;			
		std::random_shuffle(perm_buf, perm_buf + pad_input_size);
		for (size_t i = 0; i < pad_input_size; ++i)
			inv_perm_buf[perm_buf[i]] = i;
		
        cudaMemcpy(perm[p], perm_buf, sizeof(int) * pad_input_size, mode == CPU ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice);
        cudaMemcpy(inv_perm[p], inv_perm_buf, sizeof(int) * pad_input_size, mode == CPU ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice);
        
		binomial[p]->SetRandSign();
		gaussian[p]->SetRandN(mean, std);
		chisquare[p]->SetRandChi2(pad_input_size);
		chisquare[p]->Sqrt();
		chisquare[p]->Scale(1.0 / sqrt(gaussian[p]->Norm2()));
	}
    
    delete[] perm_buf;
    delete[] inv_perm_buf;
}

template<MatMode mode, typename Dtype>		
void AdaFastfoodParam<mode, Dtype>::InitializeBatch(GraphData<mode, Dtype>* g)
{
	grad_updated = false;	
}

template<MatMode mode, typename Dtype>		
void AdaFastfoodParam<mode, Dtype>::UpdateOutput(GraphData<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase)
{		
    auto* input = GetImatState(input_graph, this->operand); 
	state_input.Resize(input->rows, pad_input_size);	
	state_input.SubmatAdd(0, 0, input, 0);
	for (size_t p = 0; p < num_parts; ++p)
	{					
		// B
		state_B[p]->MulRowVec(state_input, *binomial[p]);
		// fwht
		wht_helper->Transform(input->rows, state_B[p]->data);
		state_B[p]->Scale(1.0 / sqrt(pad_input_size));
		// G
		state_G[p]->MulRowVec(*state_B[p], *gaussian[p]);        
		// Pi
		state_P[p]->ShuffleCols(*state_G[p], perm[p]);
		// fwht
		wht_helper->Transform(input->rows, state_P[p]->data);
		state_P[p]->Scale(1.0 / sqrt(pad_input_size));
		// S
		state_S[p]->MulRowVec(*state_P[p], *chisquare[p]);
	}
	state_output.Resize(input->rows, pad_output_size);
	if (num_parts == 1)
		state_output.ReduceCols(*state_S[0]);
	else  // merge 
	{
        state_output.ConcatCols(state_S);
	}
	if (bo == BiasOption::BIAS)
	{
		state_output.AddRowVec(bias, 1.0);
	}
	if (beta == 0.0)
		output->Resize(input->rows, output_size);
	output->AddSubmat(state_output, 0, 0, beta);
}

template<MatMode mode, typename Dtype>
void AdaFastfoodParam<mode, Dtype>::BackProp(DenseMat<mode, Dtype>* gradOutput)
{
    if (grad_updated)
        return;
	grad_updated = true;
	
	state_output.SubmatAdd(0, 0, *gradOutput, 0);
    
    // state_P -> state_S (grad for S)
	if (pad_output_size <= pad_input_size)
		state_S[0]->ConcatCols(state_output);
    else 
        state_output.ScatterCols(state_S);
        
    for (size_t p = 0; p < num_parts; ++p)
    {		
		// S
		state_G[p]->MulRowVec(*state_S[p], *chisquare[p]);
		// H
		state_G[p]->Scale(1.0 / sqrt(pad_input_size));		 
		wht_helper->Transform(state_G[p]->rows, state_G[p]->data);
		// Pi				
		grad_buf[p]->Resize(state_G[p]->rows, state_G[p]->cols);
		grad_buf[p]->ShuffleCols(*state_G[p], inv_perm[p]);
		// state_B -> grad_buf (grad for G)
		// G
		state_G[p]->MulRowVec(*grad_buf[p], *gaussian[p]);
		// H
		state_G[p]->Scale(1.0 / sqrt(pad_input_size));
		wht_helper->Transform(state_G[p]->rows, state_G[p]->data);
		// state_input -> state_G (grad for B)
	}
}

template<MatMode mode, typename Dtype>		
void AdaFastfoodParam<mode, Dtype>::UpdateGradInput(GraphData<mode, Dtype>* gradInput_graph, DenseMat<mode, Dtype>* gradOutput, Dtype beta)
{
    auto& gradInput = GetImatState(gradInput_graph, this->operand)->DenseDerived();
    BackProp(gradOutput);
    // now state_G is the gradient after applying B in feedforward    
    grad_input.MulRowVec(*state_G[0], *binomial[0]);
    if (num_parts > 1) 
    {        
        for (size_t p = 1; p < num_parts; ++p)
            grad_input.MulRowVec(*state_G[p], *binomial[p], 1.0);
    }
    gradInput.AddSubmat(grad_input, 0, 0, beta);
}

template<MatMode mode, typename Dtype>		
void AdaFastfoodParam<mode, Dtype>::AccDeriv(GraphData<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* gradOutput)
{
	BackProp(gradOutput);
}

template<MatMode mode, typename Dtype>		
void AdaFastfoodParam<mode, Dtype>::UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum)
{
	if (momentum > 0)
	{
		throw "not implemented";
	}
	bias_multiplier.Resize(1, state_input.rows);
	bias_multiplier.Fill(1.0);	
	bias.GeMM(bias_multiplier, state_output, Trans::N, Trans::N, -lr, 1.0);
	
	for (size_t p = 0; p < num_parts; ++p)
	{
		state_S[p]->EleWiseMul(*state_P[p]);
		chisquare[p]->GeMM(bias_multiplier, *state_S[p], Trans::N, Trans::N, -lr, (1 - l2_penalty * lr));

		grad_buf[p]->EleWiseMul(*state_B[p]);
		gaussian[p]->GeMM(bias_multiplier, *grad_buf[p], Trans::N, Trans::N, -lr, (1 - l2_penalty * lr));
		
		state_G[p]->EleWiseMul(state_input);
		binomial[p]->GeMM(bias_multiplier, *state_G[p], Trans::N, Trans::N, -lr, (1 - l2_penalty * lr));			
	} 
}

template<MatMode mode, typename Dtype>		
void AdaFastfoodParam<mode, Dtype>::Serialize(FILE* fid)
{
	IParam<mode, Dtype>::Serialize(fid);	
	
	int buf = int(bo);
	assert(fwrite(&buf, sizeof(int), 1, fid) == 1);
	assert(fwrite(&input_size, sizeof(size_t), 1, fid) == 1);		
	assert(fwrite(&output_size, sizeof(size_t), 1, fid) == 1);
	assert(fwrite(&pad_input_size, sizeof(size_t), 1, fid) == 1);
	assert(fwrite(&pad_output_size, sizeof(size_t), 1, fid) == 1);
	assert(fwrite(&num_parts, sizeof(size_t), 1, fid) == 1);
	int* perm_buf = new int[pad_input_size];
    int* inv_perm_buf = new int[pad_input_size];
    
	for (size_t i = 0; i < num_parts; ++i)
	{
		binomial[i]->Serialize(fid);
		gaussian[i]->Serialize(fid);
		chisquare[i]->Serialize(fid);
        cudaMemcpy(perm_buf, perm[i], sizeof(int) * pad_input_size, mode == CPU ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost);
        cudaMemcpy(inv_perm_buf, inv_perm[i], sizeof(int) * pad_input_size, mode == CPU ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost); 
		assert(fwrite(perm_buf, sizeof(int), pad_input_size, fid) == pad_input_size);
		assert(fwrite(inv_perm_buf, sizeof(int), pad_input_size, fid) == pad_input_size);
	}
	
	if (bo == BiasOption::BIAS)
		bias.Serialize(fid);	
        
    delete[] perm_buf;
    delete[] inv_perm_buf;        
}

template<MatMode mode, typename Dtype>		
void AdaFastfoodParam<mode, Dtype>::Deserialize(FILE* fid)
{    
	IParam<mode, Dtype>::Deserialize(fid);
	
	int buf;
	assert(fread(&buf, sizeof(int), 1, fid) == 1);
	bo = (BiasOption)buf;
	assert(fread(&input_size, sizeof(size_t), 1, fid) == 1);	
	assert(fread(&output_size, sizeof(size_t), 1, fid) == 1);
	assert(fread(&pad_input_size, sizeof(size_t), 1, fid) == 1);
	assert(fread(&pad_output_size, sizeof(size_t), 1, fid) == 1);
	assert(fread(&num_parts, sizeof(size_t), 1, fid) == 1);
	int* perm_buf = new int[pad_input_size];
    int* inv_perm_buf = new int[pad_input_size];
	for (size_t i = 0; i < num_parts; ++i)
	{
		binomial[i]->Deserialize(fid);
		gaussian[i]->Deserialize(fid);
		chisquare[i]->Deserialize(fid);
        
		assert(fread(perm_buf, sizeof(int), pad_input_size, fid) == pad_input_size);
		assert(fread(inv_perm_buf, sizeof(int), pad_input_size, fid) == pad_input_size);                    
        cudaMemcpy(perm[i], perm_buf, sizeof(int) * pad_input_size, mode == CPU ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice);        
        cudaMemcpy(inv_perm[i], inv_perm_buf, sizeof(int) * pad_input_size, mode == CPU ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice);
	}
	if (bo == BiasOption::BIAS)
		bias.Deserialize(fid);
	grad_updated = false;	
    delete[] perm_buf;
    delete[] inv_perm_buf;
}

template class AdaFastfoodParam<CPU, double>;
template class AdaFastfoodParam<CPU, float>;

template class AdaFastfoodParam<GPU, double>;
template class AdaFastfoodParam<GPU, float>;