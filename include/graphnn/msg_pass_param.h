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
class Node2NodeMsgParam : public IMessagePassParam<mode, Dtype>
{
public:
		Node2NodeMsgParam(std::string _name)
            : IMessagePassParam<mode, Dtype>(_name) {} 
            
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<MatMode mode, typename Dtype>
class Edge2NodeMsgParam : public IMessagePassParam<mode, Dtype>
{
public:
		Edge2NodeMsgParam(std::string _name)
            : IMessagePassParam<mode, Dtype>(_name) {} 
            
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<MatMode mode, typename Dtype>
class Node2EdgeMsgParam : public IMessagePassParam<mode, Dtype>
{
public:
		Node2EdgeMsgParam(std::string _name)
            : IMessagePassParam<mode, Dtype>(_name) {} 
            
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<MatMode mode, typename Dtype>
class Edge2EdgeMsgParam : public IMessagePassParam<mode, Dtype>
{
public:
		Edge2EdgeMsgParam(std::string _name)
            : IMessagePassParam<mode, Dtype>(_name) {} 
            
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<MatMode mode, typename Dtype>
class SubgraphMsgParam : public IMessagePassParam<mode, Dtype>
{
public:
		SubgraphMsgParam(std::string _name) 
            : IMessagePassParam<mode, Dtype>(_name) {}
        
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<MatMode mode, typename Dtype>
class SubgraphConcatParam : public IConstParam<mode, Dtype>
{
public:
		SubgraphConcatParam(std::string _name) 
            : IConstParam<mode, Dtype>(_name), graph(nullptr) {}

        virtual void ResetOutput(const IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output) override
        {
            assert(graph->num_nodes % graph->num_subgraph == 0);
            int node_per_graph = graph->num_nodes / graph->num_subgraph; 
            // assume the labels are consecutive
            for (int i = 0; i < node_per_graph; ++i)
                for (int j = 0; j < graph->num_subgraph; ++j)
                    assert(graph->subgraph->head[j][i] == j * node_per_graph + i); 
            
            output->Zeros(graph->num_subgraph, node_per_graph * input->cols);
        }
             				 		
		virtual void UpdateOutput(IMatrix<mode, Dtype>* input, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) override
        {
            assert(beta == 0);
            int row = output->rows, col = output->cols;            
            output->CopyFrom(input);
            // resize will not change the content in output
            output->Resize(row, col);             
        }
        
        virtual void UpdateGradInput(DenseMat<mode, Dtype>* gradInput, DenseMat<mode, Dtype>* gradOutput, Dtype beta) override
        {
            assert(gradInput->rows * gradInput->cols == gradOutput->rows * gradOutput->cols);
            gradOutput->Resize(gradInput->rows, gradInput->cols);
            
            if (beta == 0)
                gradInput->CopyFrom(*gradOutput);
            else
                gradInput->Axpby(1.0, *gradOutput, beta);
        }
        
        virtual void InitConst(void* side_info) override
        {
            this->graph = static_cast<GraphStruct*>(side_info);
        }

protected:
        GraphStruct* graph;         
};


#endif