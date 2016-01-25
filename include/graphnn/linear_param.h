#ifndef LINEAR_PARAM_H
#define LINEAR_PARAM_H

#include "iparam.h"

template<MatMode mode, typename Dtype>
class DenseMat;

template<MatMode mode, typename Dtype>
class LinearParam : public IParam<mode, Dtype>
{
public:
		LinearParam(FILE* fid);
		LinearParam(std::string _name, size_t _input_size, size_t _output_size, BiasOption _bo = BiasOption::BIAS);
		LinearParam(std::string _name, size_t _input_size, size_t _output_size, Dtype mean, Dtype std, BiasOption _bo = BiasOption::BIAS);
		
		virtual void Reset(Dtype mean, Dtype std);
		virtual void InitializeBatch(GraphData<mode, Dtype>* g) override;
		
		virtual void UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum) override;
		
		virtual void Serialize(FILE* fid) override;
		virtual void Deserialize(FILE* fid) override;
		
		virtual void UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) override;
		virtual void UpdateGradInput(DenseMat<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta) override;						
		virtual void AccDeriv(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* gradOutput) override;
		
		virtual size_t OutSize() override
		{
			return output_size;
		}
		virtual size_t InSize() override
		{
			return input_size;
		}
		DenseMat<mode, Dtype> weight;
		DenseMat<mode, Dtype> bias;
		
protected:
		BiasOption bo;
		size_t input_size, output_size;
		DenseMat<mode, Dtype> delta_weight, acc_dweight;
		DenseMat<mode, Dtype> delta_bias, acc_dbias;
		DenseMat<mode, Dtype> bias_multiplier;
};

#endif