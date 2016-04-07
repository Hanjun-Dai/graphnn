#include "nn_common.h"
#include <set>

void InitModel()
{
    init_const_dict["n2n"] = &graph;
	init_const_dict["subgraph_pool"] = &graph;
    
	const Dtype init_scale = 0.01;
	
	auto* n2nsum_param = add_const<Node2NodeMsgParam>(model, "n2n");
	auto* subgsum_param = add_const<SubgraphMsgParam>(model, "subgraph_pool");
	
    auto* w_n2l = add_diff< LinearParam >(model, "input-node-to-latent", cfg::node_dim, cfg::conv_size, 0, init_scale, BiasOption::NONE);
    auto* p_node_conv = add_diff< LinearParam >(model, "linear-node-conv", cfg::conv_size, cfg::conv_size, 0, init_scale, BiasOption::NONE);  

	auto* h1_weight = add_diff<LinearParam>(model, "h1_weight", cfg::conv_size, cfg::n_hidden, 0, init_scale);
	auto* h2_weight = add_diff<LinearParam>(model, "h2_weight", cfg::n_hidden, cfg::num_class, 0, init_scale);

	auto* node_input = cl<InputLayer>("data", gnn, {});
	auto* label_layer = cl<InputLayer>("label", gnn, {});
    auto* input_message = cl<ParamLayer>(gnn, {node_input}, {w_n2l}); 
	auto* input_potential_layer = cl<ReLULayer>(gnn, {input_message}); 

	int lv = 0;
	ILayer<mode, Dtype>* cur_message_layer = input_potential_layer;
	while (lv < cfg::max_lv)
	{	
		lv++; 
		auto* n2npool = cl<ParamLayer>(gnn, {cur_message_layer}, {n2nsum_param});

		auto* node_linear = cl<ParamLayer>(gnn, {n2npool}, {p_node_conv});

		auto* merged_linear = cl<CAddLayer>(gnn, {node_linear, input_message});  

		cur_message_layer = cl<ReLULayer>(gnn, {merged_linear}); 
	}			
	
	auto* y_potential = cl<ParamLayer>(gnn, {cur_message_layer}, {subgsum_param});

	auto* hidden = cl<ParamLayer>(gnn, {y_potential}, {h1_weight});
	
	auto* reluact_out_nn = cl<ReLULayer>(gnn, {hidden}); 
	
	auto* output = cl<ParamLayer>("output", gnn, {reluact_out_nn}, {h2_weight});
	
    cl<ClassNLLCriterionLayer>("classnll", gnn, {output, label_layer}, true);
    cl<ErrCntCriterionLayer>("errcnt", gnn, {output, label_layer});
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
