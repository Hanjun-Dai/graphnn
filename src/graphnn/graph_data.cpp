#include "graph_data.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"
#include <iostream>

template<MatMode mode, typename Dtype>
GraphData<mode, Dtype>::GraphData(MatType _mt) : mt(_mt) 
{
	if (_mt == DENSE)
	{
		node_states = new DenseMat<mode, Dtype>();
		edge_states = new DenseMat<mode, Dtype>();
	} else
	{
		node_states = new SparseMat<mode, Dtype>();
		edge_states = new SparseMat<mode, Dtype>();	
	}
	graph = new GraphStruct<int>();
}

template<MatMode mode, typename Dtype>
void GraphData<mode, Dtype>::CopyFrom(GraphData<CPU, Dtype>& g)
{
    if (mt != g.mt)
    {
        if (node_states)
            delete node_states;
        if (edge_states)
            delete edge_states;
        node_states = edge_states = nullptr;
    }
    mt = g.mt;
    if (mt == DENSE)
	{
        if (node_states == nullptr)
        {
	       node_states = new DenseMat<mode, Dtype>();
           edge_states = new DenseMat<mode, Dtype>();
        }
        node_states->DenseDerived().CopyFrom(g.node_states->DenseDerived());		
        edge_states->DenseDerived().CopyFrom(g.edge_states->DenseDerived());
	} else
	{
        if (node_states == nullptr)
        {
	       node_states = new SparseMat<mode, Dtype>();
           edge_states = new SparseMat<mode, Dtype>();
        }
		node_states->SparseDerived().CopyFrom(g.node_states->SparseDerived());
		edge_states->SparseDerived().CopyFrom(g.edge_states->SparseDerived());        	
	}
    graph = g.graph;
}

template<MatMode mode, typename Dtype>
GraphData<mode, Dtype>::~GraphData()
{
	if (graph)
		delete graph;
	delete node_states;
	delete edge_states;
}

template class GraphData<CPU, float>;
template class GraphData<CPU, double>;
template class GraphData<GPU, float>;
template class GraphData<GPU, double>;