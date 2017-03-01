#include "nn_common.h"

void InitModel()
{
	init_const_dict["n2e"] = &graph;
	init_const_dict["e2e"] = &graph;
	init_const_dict["e2n"] = &graph;	

	const Dtype init_scale = 0.05;
	
	auto* n2esum_param = add_const<Node2EdgeMsgParam>(model, "n2e");	
	auto* e2esum_param = add_const<Edge2EdgeMsgParam>(model, "e2e");	
	
	auto* w_n2l = add_diff<LinearParam>(model, "input-node-to-latent", cfg::node_dim, cfg::conv_size, 0, init_scale, BiasOption::NONE);
    auto* p_edge_conv = add_diff<LinearParam>(model, "linear-edge-conv", cfg::conv_size, cfg::conv_size, 0, init_scale, BiasOption::NONE);

	auto* e2nsum_param = add_const<Edge2NodeMsgParam>(model, "e2n");
	auto* out_params = add_diff<LinearParam>(model, "outparam", cfg::conv_size, cfg::fp_len, 0, init_scale);
		
	auto* h2_weight = add_diff<LinearParam>(model, "h2_weight", cfg::n_hidden, 1, 0, init_scale);

	IParam<mode, Dtype> *node_pool_param = nullptr, *subg_param = nullptr, *h1_weight = nullptr;
	if (cfg::max_pool)
	{	
		init_const_dict["max_pool"] = &graph;	
		node_pool_param = add_const<NodeMaxPoolParam>(model, "max_pool");
	}
	if (cfg::global_pool)
	{
		init_const_dict["subgraph_pool"] = &graph;
		subg_param = add_const<SubgraphMsgParam>(model, "subgraph_pool");
		h1_weight = add_diff<LinearParam>(model, "h1_weight", cfg::fp_len, cfg::n_hidden, 0, init_scale);
	} else {
		init_const_dict["subgraph_concat"] = &graph;	
		subg_param = add_const<SubgraphConcatParam>(model, "subgraph_concat");
		h1_weight = add_diff<LinearParam>(model, "h1_weight", cfg::fp_len * cfg::num_nodes, cfg::n_hidden, 0, init_scale);
	}

   	ILayer<mode, Dtype>* node_input = cl<InputLayer>("data", gnn, {});
	auto* label_layer = cl<InputLayer>("label", gnn, {});
    auto* input_node_linear = cl<ParamLayer>(gnn, {node_input}, {w_n2l}); 
    auto* input_message = cl<ParamLayer>(gnn, {input_node_linear}, {n2esum_param});
    auto* input_potential_layer = cl<ReLULayer>(gnn, {input_message});

    int lv = 0;
    ILayer<mode, Dtype>* cur_message_layer = input_potential_layer;
    while (lv < cfg::max_lv)
    {
    	lv++;
    	auto* e2epool = cl<ParamLayer>(gnn, {cur_message_layer}, {e2esum_param}); 

    	auto* edge_linear = cl<ParamLayer>(gnn, {e2epool}, {p_edge_conv});

    	auto* merged_linear = cl<CAddLayer>(gnn, {edge_linear, input_message}); 

    	cur_message_layer = cl<ReLULayer>(gnn, {merged_linear}); 
    }

    auto* e2npool = cl<ParamLayer>(gnn, {cur_message_layer}, {e2nsum_param}); 
    auto* hidden_msg = cl<ReLULayer>(gnn, {e2npool});
    auto* out_linear = cl<ParamLayer>(gnn, {hidden_msg}, {out_params});
	auto* reluact_fp = cl<ReLULayer>(gnn, {out_linear});	

	ILayer<mode, Dtype>* y_potential = nullptr;
	if (cfg::max_pool)
	{
		auto* out_pool = cl<ParamLayer>(gnn, {reluact_fp}, {node_pool_param});
	    y_potential = cl<ParamLayer>(gnn, {out_pool}, {subg_param});	
	} else 
	{
		y_potential = cl<ParamLayer>(gnn, {reluact_fp}, {subg_param});
	}

	auto* hidden = cl<ParamLayer>(gnn, {y_potential}, {h1_weight});

	auto* reluact_out_nn = cl<ReLULayer>(gnn, {hidden}); 
	
	auto* output = cl<ParamLayer>("output", gnn, {reluact_out_nn}, {h2_weight});
	
	cl<MSECriterionLayer>("mse", gnn, {output, label_layer});	
	cl<ABSCriterionLayer>("mae", gnn, {output, label_layer}, PropErr::N);
}

int main(int argc, const char** argv)
{
	cfg::LoadParams(argc, argv);			

	GPUHandle::Init(cfg::dev_id);	

	LoadRawData(graph_data, labels);
	LoadIndexes(cfg::train_idx_file, train_idx, labels.size());
	LoadIndexes(cfg::test_idx_file, test_idx, labels.size());

	InitModel();

    MainLoop(); 
	
	GPUHandle::Destroy();
	return 0;
}
