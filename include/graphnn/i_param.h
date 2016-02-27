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
        
        virtual bool IsDiff() = 0;
        
        virtual void UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) = 0;		
		virtual void UpdateGradInput(DenseMat<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta) = 0;
                
        virtual void ResetOutput(const IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output) = 0;
                        
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
        virtual bool IsDiff() override
        {
            return true;
        }
		virtual void AccDeriv(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* gradOutput) = 0;
                
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
            
        virtual void InitConst(void* side_info) = 0;            
        virtual bool IsDiff() override
        {
            return false;
        }            
};

#endif