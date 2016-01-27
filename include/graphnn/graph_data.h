#ifndef GRAPH_DATA_H
#define GRAPH_DATA_H

#include "imatrix.h"
#include "graph_struct.h"

enum class GraphAtt
{
	NODE = 0,
	EDGE = 1,
    NODE_EDGE = 2
};

template<MatMode mode, typename Dtype>
class GraphData
{
public:
	GraphData(MatType _mt);
	~GraphData();
	
    void CopyFrom(GraphData<CPU, Dtype>& g);
    
	IMatrix<mode, Dtype> *node_states, *edge_states;
	GraphStruct<int>* graph;

    MatType mt;
};

template<MatMode mode, typename Dtype>
inline IMatrix<mode, Dtype>* GetImatState(GraphData<mode, Dtype>* graph_data, GraphAtt att)
{
    if (att == GraphAtt::NODE)
        return graph_data->node_states;
    if (att == GraphAtt::EDGE)
        return graph_data->edge_states;
    throw "err";
} 

#endif