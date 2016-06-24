#ifndef LINEAR_PARAM_H
#define LINEAR_PARAM_H

#include "i_param.h"

template<MatMode mode, typename Dtype>
class DenseMat;

template<MatMode mode, typename Dtype>
class LinearParam : public IDiffParam<mode, Dtype>
{
public:
		LinearParam(FILE* fid)
        {
            this->Deserialize(fid);
        }
        
        LinearParam(std::string _name, size_t _input_size, size_t _output_size, BiasOption _bo = BiasOption::BIAS)
            : LinearParam<mode, Dtype>(_name, _input_size, _output_size, 0, 1.0 / sqrt(output_size), _bo) {}
        
		LinearParam(std::string _name, size_t _input_size, size_t _output_size, Dtype mean, Dtype std, BiasOption _bo = BiasOption::BIAS)
            : IDiffParam<mode, Dtype>(_name), bo(_bo)
        {            
            input_size = _input_size;
	        output_size = _output_size;
            
            this->p["weight"] = new PP<mode, Dtype>();            
            if (bo == BiasOption::BIAS)            
                this->p["bias"] = new PP<mode, Dtype>();       
            
	        Reset(mean, std);
        }
        
        virtual void ResetOutput(const IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output) override
        {
            output->Zeros(input->rows, this->p["weight"]->value.cols);
        }
        
        virtual void UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) override
        {
            auto& weight = this->p["weight"]->value;
                            
            if (input->GetMatType() == DENSE)
                output->GeMM(input->DenseDerived(), weight, Trans::N, Trans::N, 1.0, beta);
            else
                output->SparseMM(input->SparseDerived(), weight, Trans::N, Trans::N, 1.0, beta);
            
            if (bo == BiasOption::BIAS)
            {
                auto& bias = this->p["bias"]->value;
                output->AddRowVec(bias, 1.0);
            }            
        }
        
        virtual void UpdateGradInput(DenseMat<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta) override
        {
            gradInput->GeMM(*gradOutput, this->p["weight"]->value, Trans::N, Trans::T, 1.0, beta);
        }
                        
		virtual void AccDeriv(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* gradOutput) override
        {
            if (input->GetMatType() == DENSE)
                this->p["weight"]->grad.GeMM(input->DenseDerived(), *gradOutput, Trans::T, Trans::N, 1.0, 1.0);
            else
                this->p["weight"]->grad.SparseMM(input->SparseDerived(), *gradOutput, Trans::T, Trans::N, 1.0, 1.0);
            
            if (bo == BiasOption::BIAS)
            {
                bias_multiplier.Resize(1, input->rows);
                bias_multiplier.Fill(1.0);
                this->p["bias"]->grad.GeMM(bias_multiplier, *gradOutput, Trans::N, Trans::N, 1.0, 1.0);
            }           
        }
                  		
		virtual void Reset(Dtype mean, Dtype std)
        {
            this->p["weight"]->value.SetRandN(mean, std, input_size, output_size);
	        this->p["weight"]->grad.Zeros(input_size, output_size);
	        if (bo == BiasOption::BIAS)
            {
                this->p["bias"]->value.Zeros(1, output_size);
                this->p["bias"]->grad.Zeros(1, output_size);
            }
        }
		
protected:
        BiasOption bo; 
		size_t input_size, output_size;
		DenseMat<mode, Dtype> bias_multiplier;
};

#endif
