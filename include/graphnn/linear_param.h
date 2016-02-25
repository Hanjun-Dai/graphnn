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
                		
		virtual void Reset(Dtype mean, Dtype std);
		
		virtual size_t OutSize() override
		{
			return output_size;
		}
        
		virtual size_t InSize() override
		{
			return input_size;
		}
        		
protected:
        BiasOption bo; 
		size_t input_size, output_size;
		DenseMat<mode, Dtype> bias_multiplier;
};

#endif