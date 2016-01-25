#ifndef NODE_POOL_PARAM_H
#define NODE_POOL_PARAM_H

#include "iparam.h"
#include "sparse_matrix.h"

template<MatMode mode, typename Dtype>
class NodePoolParam : public IParam<mode, Dtype>
{
public:
		NodePoolParam(std::string _name, PoolingOp _op, NodePoolType _npt);
		
		virtual void InitializeBatch(GraphData<mode, Dtype>* g) override;
		
		virtual void UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum) override;
		
		virtual void Serialize(FILE* fid) override;
		virtual void Deserialize(FILE* fid) override;
		
		virtual void UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) override {}
		virtual void UpdateGradInput(DenseMat<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta) override {}						
		virtual void AccDeriv(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* gradOutput) override {}
		virtual size_t OutSize() override
		{
			return 0;
		}
		virtual size_t InSize() override
		{
			return 0;
		}
		SparseMat<mode, Dtype> weight;
		PoolingOp op;
		NodePoolType npt;
        
protected:
        virtual void InitCPUWeight(GraphData<mode, Dtype>* g);
        SparseMat<CPU, Dtype>* cpu_weight;
};

template<MatMode mode, typename Dtype>
class SubgraphPoolParam : public NodePoolParam<mode, Dtype>
{
public:
		SubgraphPoolParam(std::string _name, PoolingOp _op, NodePoolType _npt);		
        
protected:
        virtual void InitCPUWeight(GraphData<mode, Dtype>* g) override;
};

#endif
