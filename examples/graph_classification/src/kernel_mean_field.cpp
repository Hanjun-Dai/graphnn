#include "nn_common.h"
#include <set>

using namespace gnn;

void InitModel()
{
	const Dtype init_scale = 0.01;
	auto graph = add_const< GraphVar >(fg, "graph", true);

	auto n2nsum_param = af< Node2NodeMsgPass<mode, Dtype> >(fg, {graph});
	auto subgsum_param = af< SubgraphMsgPass<mode, Dtype> >(fg, {graph});

	auto w_n2l = add_diff<DTensorVar>(model, "input-node-to-latent", {(uint)cfg::node_dim, cfg::conv_size});	
	auto p_node_conv = add_diff< DTensorVar >(model, "linear-node-conv", {cfg::conv_size, cfg::conv_size});
	auto h1_weight = add_diff<DTensorVar>(model, "h1_weight", {cfg::conv_size, cfg::n_hidden});
	auto h2_weight = add_diff<DTensorVar>(model, "h2_weight", {cfg::n_hidden, (uint)cfg::num_class});

	w_n2l->value.SetRandN(0, init_scale);
	p_node_conv->value.SetRandN(0, init_scale);
	h1_weight->value.SetRandN(0, init_scale);
	h2_weight->value.SetRandN(0, init_scale);
    fg.AddParam(w_n2l);
    fg.AddParam(p_node_conv);
    fg.AddParam(h1_weight);
    fg.AddParam(h2_weight);
    
	auto node_input = add_const< DTensorVar<mode, Dtype> >(fg, "data", true);
	auto label = add_const< SpTensorVar<mode, Dtype> >(fg, "label", true);

	auto input_message = af<MatMul>(fg, {node_input, w_n2l});
	auto input_potential_layer = af<ReLU>(fg, {input_message}); 
	int lv = 0;
	auto cur_message_layer = input_potential_layer;
	while (lv < cfg::max_lv)
	{
		lv++;
		auto n2npool = af<MatMul>(fg, {n2nsum_param, cur_message_layer});
		auto node_linear = af<MatMul>(fg, {n2npool, p_node_conv});
		auto merged_linear = af<ElewiseAdd>(fg, {node_linear, input_message});
		cur_message_layer = af<ReLU>(fg, {merged_linear}); 
	}

	auto y_potential = af<MatMul>(fg, {subgsum_param, cur_message_layer});
	auto hidden = af<MatMul>(fg, {y_potential, h1_weight});
	auto reluact_out_nn = af<ReLU>(fg, {hidden}); 
	auto output = af< MatMul >(fg, {reluact_out_nn, h2_weight});

	auto ce = af< CrossEntropy >(fg, {output, label}, true);
	auto loss = af< ReduceMean >(fg, {ce});

	auto truth = af< ArgMax >(fg, {label});
    auto cmp = af< InTopK<mode, Dtype> >(fg, std::make_pair(output, truth));
    auto real_cmp = af< TypeCast<mode, Dtype> >(fg, {cmp});

	auto acc = af< ReduceMean >(fg, { real_cmp });	

	targets.clear();
	targets.push_back(loss);
	targets.push_back(acc);

	objs.clear();
	objs.push_back(loss);
}

int main(int argc, const char** argv)
{
	cfg::LoadParams(argc, argv);			
	GpuHandle::Init(cfg::dev_id, 1);

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

    GpuHandle::Destroy();
	return 0;
}