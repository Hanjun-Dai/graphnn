#ifndef MSG_PASS_PARAM_H
#define MSG_PASS_PARAM_H

#include "i_param.h"
#include "graph_struct.h"
#include "sparse_matrix.h"

template<MatMode mode, typename Dtype>
class IMessagePassParam : public IConstParam<mode, Dtype>
{
public:
		IMessagePassParam(std::string _name);
		
        virtual void InitConst(void* side_info) override
        {
            this->InitCPUWeight(static_cast<GraphStruct*>(side_info));
            if (mode == GPU)
                this->weight.CopyFrom(*(this->cpu_weight));
        }
        		
        virtual void ResetOutput(const IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output) override
        {
            output->Resize(this->weight.rows, input->cols);
        }
             				 		
		virtual void UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) override
        {
            auto& prev_states = input->DenseDerived();
            output->SparseMM(this->weight, prev_states, Trans::N, Trans::N, 1.0, beta);                
        }
        
        virtual void UpdateGradInput(DenseMat<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta) override
        {
            auto& prev_grad = gradInput->DenseDerived();
            prev_grad.SparseMM(this->weight, *gradOutput, Trans::T, Trans::N, 1.0, beta);
        }
        
		SparseMat<mode, Dtype> weight;
		
protected:
        virtual void InitCPUWeight(GraphStruct* graph) = 0;
        SparseMat<CPU, Dtype>* cpu_weight;
};

template<MatMode mode, typename Dtype>
class Node2NodePoolParam : public IMessagePassParam<mode, Dtype>
{
public:
		Node2NodePoolParam(std::string _name)
            : IMessagePassParam<mode, Dtype>(_name) {} 
            
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<MatMode mode, typename Dtype>
class Edge2NodePoolParam : public IMessagePassParam<mode, Dtype>
{
public:
		Edge2NodePoolParam(std::string _name)
            : IMessagePassParam<mode, Dtype>(_name) {} 
            
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<MatMode mode, typename Dtype>
class Node2EdgePoolParam : public IMessagePassParam<mode, Dtype>
{
public:
		Node2EdgePoolParam(std::string _name)
            : IMessagePassParam<mode, Dtype>(_name) {} 
            
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<MatMode mode, typename Dtype>
class Edge2EdgePoolParam : public IMessagePassParam<mode, Dtype>
{
public:
		Edge2EdgePoolParam(std::string _name)
            : IMessagePassParam<mode, Dtype>(_name) {} 
            
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<MatMode mode, typename Dtype>
class SubgraphPoolParam : public IMessagePassParam<mode, Dtype>
{
public:
		SubgraphPoolParam(std::string _name) 
            : IMessagePassParam<mode, Dtype>(_name) {}
        
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

#endif