#ifndef BATCH_NORM_PARAM_H
#define BATCH_NORM_PARAM_H

#include "iparam.h"

template<MatMode mode, typename Dtype>
class DenseMat;

template<MatMode mode, typename Dtype>
class BatchNormParam : public IParam<mode, Dtype>
{
public:
    BatchNormParam(FILE* fid);     
    BatchNormParam(std::string _name, GraphAtt _operand, size_t _input_size, bool _parametrized, Dtype _eps = 1e-5, Dtype _smooth = 0.1);
    
    virtual void InitializeBatch(GraphData<mode, Dtype>* g) override { cur_grad_output = nullptr; }

	virtual void UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum) override;
    
    virtual void Serialize(FILE* fid) override;
	virtual void Deserialize(FILE* fid) override;
    
    virtual void UpdateOutput(GraphData<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) override;
	virtual void UpdateGradInput(GraphData<mode, Dtype>* gradInput_graph, DenseMat<mode, Dtype>* gradOutput, Dtype beta) override;						
	virtual void AccDeriv(GraphData<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* gradOutput) override;
    
	virtual size_t OutSize() override
	{
		return input_size;
	}
	virtual size_t InSize() override
	{
		return input_size;
	}        
    
    size_t input_size;
    bool parametrized;
    Dtype eps, smooth;
    DenseMat<mode, Dtype> acc_mean, acc_inv_std;
    DenseMat<mode, Dtype> row_buffer, mat_buffer, normed_output, cur_inv_std;
    DenseMat<mode, Dtype> scale, bias, row_multiplier;
    DenseMat<mode, Dtype>* cur_grad_output; 
};

#endif
