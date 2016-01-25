#ifndef GRAPH_DATA_H
#define GRAPH_DATA_H

#include "imatrix.h"
#include "graph_struct.h"

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

#endif