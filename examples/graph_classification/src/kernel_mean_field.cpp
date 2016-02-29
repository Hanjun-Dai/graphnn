#include "nn_common.h"
#include <set>

void InitModel()
{
    init_const_dict["n2n"] = &graph;
	init_const_dict["e2n"] = &graph;
	init_const_dict["subgraph_pool"] = &graph;
/*    
	const Dtype init_scale = 0.01;
	
	auto* n2nsum_param = new NodeCentricPoolParam<mode, Dtype>("n2n");	
	gnn.AddParam(n2nsum_param);	
	auto* subgsum_param = new SubgraphPoolParam<mode, Dtype>("subgraph_pool");
	gnn.AddParam(subgsum_param);
	
    auto* w_n2l = new LinearParam<mode, Dtype>("input-node-to-latent", cfg::node_dim, cfg::conv_size, 0, init_scale, BiasOption::NONE);		
    gnn.AddParam(w_n2l);
    auto* p_node_conv = new LinearParam<mode, Dtype>("linear-node-conv", cfg::conv_size, cfg::conv_size, 0, init_scale, BiasOption::NONE);		
    gnn.AddParam(p_node_conv);

	ILayer<mode, Dtype>* input_layer = new InputLayer<mode, Dtype>("input", GraphAtt::NODE);
    auto* input_message = new SingleParamNodeLayer<mode, Dtype>("input_node_linear", w_n2l, GraphAtt::NODE); 
	auto* input_potential_layer = new ReLULayer<mode, Dtype>("init_relu", GraphAtt::NODE, WriteType::INPLACE); 

	gnn.AddEdge(input_layer, input_message);
	gnn.AddEdge(input_message, input_potential_layer);

	int lv = 0;
	ILayer<mode, Dtype>* cur_message_layer = input_potential_layer;
	while (lv < cfg::max_lv)
	{	
		lv++; 
		auto* n2npool = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("n2npool_%d", lv), n2nsum_param, GraphAtt::NODE);

		auto* node_linear = new SingleParamNodeLayer<mode, Dtype>(fmt::sprintf("nodelinear_%d", lv), p_node_conv, GraphAtt::NODE);

		auto* merged_linear = new NodeGatherLayer<mode, Dtype>(fmt::sprintf("message_%d", lv));  

		auto* new_message = new ReLULayer<mode, Dtype>(fmt::sprintf("relu_%d", lv), GraphAtt::NODE, WriteType::INPLACE); 

		gnn.AddEdge(cur_message_layer, n2npool);
		gnn.AddEdge(n2npool, node_linear);		
		gnn.AddEdge(node_linear, merged_linear);	
		gnn.AddEdge(input_message, merged_linear);
		gnn.AddEdge(merged_linear, new_message);

		cur_message_layer = new_message;
	}			
	
	auto* y_potential = new SingleParamNodeLayer<mode, Dtype>("y_potential", subgsum_param, GraphAtt::NODE);
	gnn.AddLayer(y_potential);
	gnn.AddEdge(cur_message_layer, y_potential);


	auto* h1_weight = new LinearParam<mode, Dtype>("h1_weight", cfg::conv_size, cfg::n_hidden, 0, init_scale);
	gnn.AddParam(h1_weight); 

	auto* hidden = new SingleParamNodeLayer<mode, Dtype>("hidden", h1_weight, GraphAtt::NODE);
	gnn.AddEdge(y_potential, hidden);	
	
	auto* reluact_out_nn = new ReLULayer<mode, Dtype>("reluact_out_nn", GraphAtt::NODE, WriteType::INPLACE); 
	gnn.AddEdge(hidden, reluact_out_nn);		
		
	auto* h2_weight = new LinearParam<mode, Dtype>("h2_weight", cfg::n_hidden, cfg::num_class, 0, init_scale);
	gnn.AddParam(h2_weight);
	auto* output = new SingleParamNodeLayer<mode, Dtype>("output", h2_weight, GraphAtt::NODE);
	gnn.AddEdge(reluact_out_nn, output);
	
    auto* classnll = new ClassNLLCriterionLayer<mode, Dtype>("classnll", true);
    auto* errcnt = new ErrCntCriterionLayer<mode, Dtype>("errcnt");

	gnn.AddEdge(output, classnll);
	gnn.AddEdge(output, errcnt);
*/    
}

int main(int argc, const char** argv)
{
	cfg::LoadParams(argc, argv);			

	GPUHandle::Init(cfg::dev_id);	

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
	
	GPUHandle::Destroy();
    
	return 0;
}
