#ifndef MSG_PASS_PARAM_H
#define MSG_PASS_PARAM_H

#include "iparam.h"
#include "sparse_matrix.h"

template<MatMode mode, typename Dtype>
class IMessagePassParam : public IParam<mode, Dtype>
{
public:
		IMessagePassParam(std::string _name, GraphAtt _operand);
		
		virtual void InitializeBatch(GraphData<mode, Dtype>* g) override;
		
		virtual void UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum) override {}
		
		virtual void Serialize(FILE* fid) override;
		virtual void Deserialize(FILE* fid) override;
        
		virtual void UpdateOutput(GraphData<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) override;
        virtual void UpdateGradInput(GraphData<mode, Dtype>* gradInput_graph, DenseMat<mode, Dtype>* gradOutput, Dtype beta) override;
        
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
        virtual void InitCPUWeight(GraphData<mode, Dtype>* g) = 0;
        SparseMat<CPU, Dtype>* cpu_weight;
};

template<MatMode mode, typename Dtype>
class NodePoolParam : public IMessagePassParam<mode, Dtype>
{
public:
		NodePoolParam(std::string _name, GraphAtt _operand);        
protected:
        virtual void InitCPUWeight(GraphData<mode, Dtype>* g) override;
};

template<MatMode mode, typename Dtype>
class SubgraphPoolParam : public IMessagePassParam<mode, Dtype>
{
public:
		SubgraphPoolParam(std::string _name, GraphAtt _operand);		
        
protected:
        virtual void InitCPUWeight(GraphData<mode, Dtype>* g) override;
};

#endif
