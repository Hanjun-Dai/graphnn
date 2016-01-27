#ifndef IPARAM_H
#define IPARAM_H

#include "imatrix.h"
#include "graph_data.h"
#include <string>

enum class BiasOption
{
	NONE,
	BIAS
};

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
			int buf = name.size();
			assert(fwrite(&buf, sizeof(int), 1, fid) == 1);	
			assert(fwrite(name.c_str(), sizeof(char), buf, fid) == buf);
		}
		
		virtual void Deserialize(FILE* fid)
		{
			assert(fread(&batch_prepared, sizeof(bool), 1, fid) == 1);
            int buf;
			assert(fread(&buf, sizeof(int), 1, fid) == 1);
			char char_buf[buf + 1];
			assert(fread(char_buf, sizeof(char), buf, fid) == buf);
			char_buf[buf] = '\0';
			name = std::string(char_buf);
		}
		
		virtual void InitializeBatch(GraphData<mode, Dtype>* g, GraphAtt operand) = 0;
		
		virtual void UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum) {}		
		
		virtual void UpdateOutput(IMatrix<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) {}
		virtual void UpdateGradInput(IMatrix<mode, Dtype>* gradInput_graph, DenseMat<mode, Dtype>* gradOutput, Dtype beta) {}						
		virtual void AccDeriv(IMatrix<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* gradOutput) {}
		virtual size_t OutSize() { throw "not implemented"; }
		virtual size_t InSize() { throw "not implemented"; }
        
		std::string name;
		bool batch_prepared;
};

#endif