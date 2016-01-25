#ifndef NONSHARED_LINEAR_PARAM_H
#define NONSHARED_LINEAR_PARAM_H

#include "linear_param.h"

template<MatMode mode, typename Dtype>
class DenseMat;

template<MatMode mode, typename Dtype>
class NonsharedLinearParam : public LinearParam<mode, Dtype>
{
public:
		NonsharedLinearParam(FILE* fid);
		NonsharedLinearParam(std::string _name, size_t _input_size, size_t _output_size, BiasOption _bo = BiasOption::BIAS);
		NonsharedLinearParam(std::string _name, size_t _input_size, size_t _output_size, Dtype mean, Dtype std, BiasOption _bo = BiasOption::BIAS);
		
		virtual void Reset(Dtype mean, Dtype std) override;
		virtual void InitializeBatch(GraphData<mode, Dtype>* g) override;
		
		virtual void AccDeriv(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* gradOutput) override;
		virtual void UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum) override;

private:
		IMatrix<mode, Dtype>* cur_input;
		DenseMat<mode, Dtype>* cur_gradOutput;					
};

#endif