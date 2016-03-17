#ifndef GRAPH_POOL_PARAM_H
#define GRAPH_POOL_PARAM_H

#include "msg_pass_param.h"
#include "graph_struct.h"
#include "sparse_matrix.h"

template<MatMode mode, typename Dtype>
class NodeAvgPoolParam : public IMessagePassParam<mode, Dtype>
{
public:
		NodeAvgPoolParam(std::string _name)
            : IMessagePassParam<mode, Dtype>(_name) {} 
            
protected:
        virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<MatMode mode, typename Dtype>
class NodeMaxPoolParam; 

template<typename Dtype>
class NodeMaxPoolParam<CPU, Dtype> : public IConstParam<CPU, Dtype>
{
public:
        NodeMaxPoolParam(std::string _name)
            : IConstParam<CPU, Dtype>(_name) { max_index.clear(); }

        virtual void InitConst(void* side_info) override
        {
            graph = static_cast<GraphStruct*>(side_info); 
        }
        		
        virtual void ResetOutput(const IMatrix<CPU, Dtype>* input, DenseMat<CPU, Dtype>* output) override
        {
            output->Zeros(graph->num_nodes, input->cols);
        }
             				 		
		virtual void UpdateOutput(IMatrix<CPU, Dtype>* input, DenseMat<CPU, Dtype>* output, Dtype beta, Phase phase) override;
        
        virtual void UpdateGradInput(DenseMat<CPU, Dtype>* gradInput, DenseMat<CPU, Dtype>* gradOutput, Dtype beta) override;                     
protected:
        GraphStruct* graph;
        std::vector<int> max_index;                
};

template<typename Dtype>
class NodeMaxPoolParam<GPU, Dtype> : public IConstParam<GPU, Dtype>
{
public:
    
};

#endif