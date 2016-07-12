#ifndef CONST_SCALAR_PARAM
#define CONST_SCALAR_PARAM

#include "i_param.h"

template<MatMode mode, typename Dtype>
class ConstScalarParam : public IConstParam<mode, Dtype>
{
public:
    ConstScalarParam(std::string _name, Dtype _a, Dtype _b)
        : IConstParam<mode, Dtype>(_name), a(_a), b(_b) {}
    
    virtual void InitConst(void* side_info) override {}
       
    virtual void ResetOutput(const IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output) override
    {
        output->Resize(input->rows, input->cols);        
    }             
    
    virtual void UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) override
    {
        output->Axpby(a, input->DenseDerived(), beta);
        output->Add(b);    
    }
    
    virtual void UpdateGradInput(DenseMat<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta) override
    {
        gradInput->Axpby(a, *gradOutput, beta);    
    }
    
    const Dtype a, b;
};

#endif
