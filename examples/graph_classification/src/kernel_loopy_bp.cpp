#include "nn_common.h"
#include <set>

void InitModel()
{

}

int main(int argc, const char** argv)
{
	cfg::LoadParams(argc, argv);			

	LoadIndexes(cfg::train_idx_file, train_idx);
	LoadIndexes(cfg::test_idx_file, test_idx);
	LoadRawData(graph_data, labels);

	cfg::node_dim = 0;
	for (size_t i = 0; i < graph_data.size(); ++i)
	{
		auto& g = graph_data[i];
		for (int j = 0; j < g.num_nodes; ++j)
		{
			if (g.node_label[j] > cfg::node_dim)
				cfg::node_dim = g.node_label[j];
		}
	}
	std::cerr << "dimension of node feature: " << cfg::node_dim << std::endl;

	InitModel();

    MainLoop(); 
	    
	return 0;
}