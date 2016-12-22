#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include "util/graph_struct.h"

using namespace gnn;

struct Graph
{
	Graph() {}

	Graph(int _num_nodes) : num_nodes(_num_nodes)
	{
		adj.Resize(num_nodes); 
		node_label.clear();
	}

	int num_nodes; 
	LinkedTable<int> adj;
	std::vector<int> node_label; 
};

inline void LoadIndexes(const char* filename, std::vector<int>& idx_list)
{
    std::cerr << "loading indexes " << filename << std::endl;
	idx_list.clear();
	FILE* fid = fopen(filename, "r");
	int idx;
	while (fscanf(fid, "%d", &idx) != EOF)
	{
		idx_list.push_back(idx);
	}
	fclose(fid);
}

inline void LoadRawData(std::vector< Graph >& graph_data, std::vector<int>& labels)
{
	graph_data.clear();
	labels.clear();
	std::ifstream ff(cfg::graph_file);

	int num_graph;
	ff >> num_graph;

	int num_nodes, num_neighbor, l;
	for (int i = 0; i < num_graph; ++i)
	{
		ff >> num_nodes >> l;
		labels.push_back(l);
		if (l + 1 > cfg::num_class)
			cfg::num_class = l + 1;
		Graph g(num_nodes);
		for (int j = 0; j < num_nodes; ++j)
		{
			ff >> l >> num_neighbor;
			g.node_label.push_back(l);
			for (int k = 0; k < num_neighbor; ++k)
			{
				ff >> l;
				g.adj.AddEntry(j, l); 		
			}
		}
		graph_data.push_back(g);
	}
}



#endif
