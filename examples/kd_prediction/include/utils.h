#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <string>
#include "graph_struct.h"

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

inline void LoadIndexes(const char* filename, std::vector<int>& idx_list, size_t num_samples)
{
	if (filename == nullptr)
	{
		std::cerr << num_samples << std::endl;
		std::cerr << "loading all" << std::endl;
		idx_list.resize(num_samples);
		for (size_t i = 0; i < idx_list.size(); ++i)
			idx_list[i] = i;
		std::random_shuffle(idx_list.begin(), idx_list.end());
		return;
	}
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

inline int Str2Id(const std::string& st)
{
	int id = 0;
	for (size_t i = 0; i < st.size(); ++i)
	{
		int t = 0;
		switch (st[i])
		{
			case 'A':
				t = 0;
				break;
			case 'T':
				t = 1;
				break;
			case 'C':
				t = 2;
				break;
			case 'G':
				t = 3; 
				break;
			default:
				t = 4;
				break;
		}
		id = id * (4 + cfg::pad) + t;
	}
	return id;
}

inline Graph* BuildGraph(std::string st)
{
	if (cfg::pad)
	{
		cfg::num_nodes = st.size();
	}
	else
		cfg::num_nodes = st.size() - cfg::window_size + 1;

	Graph* g = new Graph(cfg::num_nodes);

	for (int j = 0; j < (int)cfg::num_nodes; ++j)
	{
		std::string buf = "";
		if (cfg::pad)
		{				
			for (int t = j - cfg::window_size / 2; t < j - cfg::window_size / 2 + cfg::window_size; ++t)
			{
				if (t < 0)
					buf = buf + " ";
				else if (t >= (int)st.size())
					buf = buf + " ";
				else
					buf = buf + st[t];
			}
		} else 
			buf = st.substr(j, cfg::window_size);

		g->node_label.push_back(Str2Id(buf));
				
		if (j)
			g->adj.AddEntry(j, j - 1);
		if (j < cfg::num_nodes - 1)
			g->adj.AddEntry(j, j + 1);	
	}
	return g;
}

inline void LoadRawData(std::vector< Graph >& graph_data, std::vector<Dtype>& labels)
{
	graph_data.clear();
	labels.clear();
	std::ifstream ff(cfg::string_file);

	int num_graph;
	ff >> num_graph;

	std::string st;
	Dtype l;
	for (int i = 0; i < num_graph; ++i)
	{
		ff >> l >> st;
		labels.push_back(l * cfg::scale);
		
		Graph* g = BuildGraph(st);		
		graph_data.push_back(*g);
	}
}


#endif
