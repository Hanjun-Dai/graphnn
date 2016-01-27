#ifndef MSG_PASS_PARAM_H
#define MSG_PASS_PARAM_H

#include "iparam.h"
#include "sparse_matrix.h"

template<MatMode mode, typename Dtype>
class IMessagePassParam : public IParam<mode, Dtype>
{
public:
		IMessagePassParam(std::string _name);
		
		virtual void InitializeBatch(GraphData<mode, Dtype>* g, GraphAtt operand) override;
		
		virtual void UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum) override {}
		
		virtual void Serialize(FILE* fid) override;
		virtual void Deserialize(FILE* fid) override;
        
		virtual void UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) override;
        virtual void UpdateGradInput(IMatrix<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta) override;
        
		virtual size_t OutSize() override
		{
			return 0;
		}
		virtual size_t InSize() override
		{
			return 0;
		}
		SparseMat<mode, Dtype> weight;
		
protected:
        virtual void InitCPUWeight(GraphData<mode, Dtype>* g, GraphAtt operand) = 0;
        SparseMat<CPU, Dtype>* cpu_weight;
};

template<MatMode mode, typename Dtype>
class NodeCentricPoolParam : public IMessagePassParam<mode, Dtype>
{
public:
		NodeCentricPoolParam(std::string _name)
            : IMessagePassParam<mode, Dtype>(_name) {} 
            
protected:
        virtual void InitCPUWeight(GraphData<mode, Dtype>* g, GraphAtt operand) override;
};

template<MatMode mode, typename Dtype>
class SubgraphPoolParam : public IMessagePassParam<mode, Dtype>
{
public:
		SubgraphPoolParam(std::string _name) 
            : IMessagePassParam<mode, Dtype>(_name) {}
        
protected:
        virtual void InitCPUWeight(GraphData<mode, Dtype>* g, GraphAtt operand) override;
};

#endif
