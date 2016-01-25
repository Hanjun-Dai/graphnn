#ifndef IPARAM_H
#define IPARAM_H

#include "imatrix.h"
#include <string>

enum class PoolingOp
{
	SUM = 0, 
	AVG = 1, 
	MAX = 2
};

enum class BiasOption
{
	NONE,
	BIAS
};

enum class NodePoolType
{
	N2N = 0,
	E2N = 1	
};

template<MatMode mode, typename Dtype>
class GraphData;

template<MatMode mode, typename Dtype>
class IParam
{
public:
		IParam() {} 
		
		IParam(std::string _name)
		{
			name = _name;
			batch_prepared = false;
		}
		
		virtual void Serialize(FILE* fid) 
		{
			assert(fwrite(&batch_prepared, sizeof(bool), 1, fid) == 1);
			int len = name.size();
			assert(fwrite(&len, sizeof(int), 1, fid) == 1);	
			assert(fwrite(name.c_str(), sizeof(char), len, fid) == len);
		}
		
		virtual void Deserialize(FILE* fid)
		{
			assert(fread(&batch_prepared, sizeof(bool), 1, fid) == 1);
			int len;
			assert(fread(&len, sizeof(int), 1, fid) == 1);
			char buf[len + 1];
			assert(fread(buf, sizeof(char), len, fid) == len);
			buf[len] = '\0';
			name = std::string(buf);
		}
		
		virtual void InitializeBatch(GraphData<mode, Dtype>* g) = 0;
		
		virtual void UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum) {}		
		
		virtual void UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) {}
		virtual void UpdateGradInput(DenseMat<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta) {}						
		virtual void AccDeriv(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* gradOutput) {}
		virtual size_t OutSize() { throw "not implemented"; }
		virtual size_t InSize() { throw "not implemented"; }
		std::string name;
		bool batch_prepared;
};

#endif