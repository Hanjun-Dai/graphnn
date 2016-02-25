#ifndef IPARAM_H
#define IPARAM_H

#include "imatrix.h"
#include <string>
#include <stdexcept>

template<MatMode mode, typename Dtype>
class IParam
{
public:
		IParam()
		{
            value_dict.clear();
            grad_dict.clear();
		}
		
		virtual void Serialize(FILE* fid) 
		{			
		}
		
		virtual void Deserialize(FILE* fid)
		{			
                        
		}		
		//virtual void UpdateOutput(IMatrix<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) {}
		//virtual void UpdateGradInput(IMatrix<mode, Dtype>* gradInput_graph, DenseMat<mode, Dtype>* gradOutput, Dtype beta) {}						
		//virtual void AccDeriv(IMatrix<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* gradOutput) {}
                
		virtual size_t OutSize() 
        { 
            throw std::runtime_error("not implemented");
        }
        
		virtual size_t InSize() 
        { 
            throw "not implemented"; 
        }
        
        std::map< std::string, IMatrix<mode, Dtype>* > value_dict, grad_dict;        
};

#endif