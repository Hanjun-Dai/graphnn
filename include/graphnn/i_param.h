#ifndef IPARAM_H
#define IPARAM_H

#include "dense_matrix.h"
#include <string>
#include <map>

enum class BiasOption
{
	NONE,
	BIAS
};

template<MatMode mode, typename Dtype>
struct PP
{
        PP(){}
        
        PP(size_t rows, size_t cols)
        {
            value.Resize(rows, cols);
            grad.Resize(rows, cols);
        }
        
        DenseMat<mode, Dtype> value, grad;
};

template<MatMode mode, typename Dtype>
class IParam
{
public:
        IParam() {}
		IParam(std::string _name)
            : name(_name)
		{
		}
		
		virtual void Serialize(FILE* fid) 
		{			
		}
		
		virtual void Deserialize(FILE* fid)
		{			                        
		}
        
        virtual void UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Phase phase) = 0;		
		//virtual void UpdateGradInput(IMatrix<mode, Dtype>* gradInput_graph, DenseMat<mode, Dtype>* gradOutput, Dtype beta) {}						
		//virtual void AccDeriv(IMatrix<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* gradOutput) {}
                
		virtual size_t OutSize() 
        { 
            throw std::runtime_error("not implemented");
        }
        
		virtual size_t InSize() 
        { 
            throw std::runtime_error("not implemented"); 
        }
        
        std::string name;        
};

template<MatMode mode, typename Dtype>
class IDiffParam : public IParam<mode, Dtype>
{
public:  
        IDiffParam() {}
        IDiffParam(std::string _name)
            : IParam<mode, Dtype>(_name)
            {
                p.clear();
            }
            
        std::map<std::string, PP<mode, Dtype>*> p;
};

template<MatMode mode, typename Dtype>
class IConstParam : public IParam<mode, Dtype>
{
public:
        IConstParam() {}
        IConstParam(std::string _name)
            : IParam<mode, Dtype>(_name)
            {
                
            }    
};

#endif