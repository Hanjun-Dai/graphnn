#include "graph_struct.h"
#include <cstring>
#include <iostream>
#include <string>

template<typename T>
GraphStruct<T>::GraphStruct()
{
		adj_list = new LinkedTable< std::pair<int, int> >();
		subgraph = new LinkedTable< int >();
		num_nodes = num_edges = num_subgraph = 0;
		degree_in.clear();
		idx_map.clear();
		t_map.clear();
}

template<typename T>
GraphStruct<T>::~GraphStruct()
{
		delete adj_list;
		delete subgraph;
}

template<typename T>
void GraphStruct<T>::AddEdge(int idx, T node_x, T node_y)
{
		if (idx_map.count(node_x) == 0)
			AddNode(0, node_x);
		if (idx_map.count(node_y) == 0)
			AddNode(0, node_y);
		int x = idx_map[node_x], y = idx_map[node_y];
		
		adj_list->AddEntry(x, std::pair<int, int>(idx, y));
		num_edges++;
		degree_in[y]++;
}

template<typename T>
void GraphStruct<T>::AddNode(int subg_id, T node)
{
		if (idx_map.count(node))
			return;
		
		int cur_idx = idx_map.size();
		idx_map[node] = cur_idx;
		t_map[cur_idx] = node;	
		subgraph->AddEntry(subg_id, cur_idx);
		
		if (cur_idx == num_nodes)
		{
			num_nodes++;
			degree_in.push_back(0);
		}
}

template<typename T>
void GraphStruct<T>::Resize(unsigned _num_subgraph, unsigned _num_nodes)
{
		num_nodes = _num_nodes;
		num_edges = 0;
		num_subgraph = _num_subgraph;
		
		adj_list->Resize(num_nodes);
		subgraph->Resize(num_subgraph);
		degree_in.resize(num_nodes);
		for (size_t i = 0; i < degree_in.size(); ++i)
			degree_in[i] = 0;
}

template<typename T>
void GraphStruct<T>::TopSort(std::vector<std::pair<T, T> >& sorted_edges)
{
		int* next_idx = new int[num_nodes];
		memcpy(next_idx, degree_in.data(), sizeof(int) * num_nodes);
		
		//sorted_list.clear();
		sorted_edges.clear();
		int top = -1, x, y;
		for (int i = 0; i < num_nodes; ++i)
			if (next_idx[i] == 0)
			{				
				next_idx[i] = top;
				top = i;
			}
		while (top >= 0)
		{
			x = top;
			//sorted_list.push_back(t_map[x]);
			top = next_idx[top];
			if (adj_list->head.size() > x)
			{
				for (size_t i = 0; i < adj_list->head[x].size(); ++i)
				{
					y = adj_list->head[x][i].second;
					sorted_edges.push_back(std::pair<T, T>(t_map[x], t_map[y]));
					next_idx[y]--;
					if (next_idx[y] == 0)
					{
						next_idx[y] = top;
						top = y;
					}
				}
			}
		}
		
		delete[] next_idx;
}

template class GraphStruct<std::string>;