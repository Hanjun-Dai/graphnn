#ifndef NN_COMMON_H
#define NN_COMMON_H

#include "config.h"
#include "utils.h"
#include "dense_matrix.h"
#include "linear_param.h"
#include "nngraph.h"
#include "msg_pass_param.h"
#include "param_layer.h"
#include "input_layer.h"
#include "cppformat/format.h"
#include "relu_layer.h"
#include "c_add_layer.h"
#include "learner.h"
#include "model.h"
#include "classnll_criterion_layer.h"
#include "err_cnt_criterion_layer.h"
const MatMode mode = CPU;

std::vector< Graph > graph_data;
std::vector<int> labels;
std::vector<int> train_idx, test_idx;

NNGraph<mode, Dtype> gnn;
Model<mode, Dtype> model;
MomentumSGDLearner<mode, Dtype>* learner;
std::map<std::string, void*> init_const_dict;
DenseMat<CPU, Dtype> x_cpu;
SparseMat<CPU, Dtype> y_cpu;
GraphStruct graph;

DenseMat<mode, Dtype> input;
SparseMat<mode, Dtype> label;

std::vector< unsigned > prefix_sum;
inline void GetBatch(const std::vector<int>& idx_list, unsigned st, unsigned num)
{
	unsigned ed = idx_list.size() < st + num ? idx_list.size() : st + num;
	num = ed - st;		
	prefix_sum.clear();

	unsigned node_cnt = 0;
	for (unsigned i = st; i < ed; ++i)
	{
		auto& g = graph_data[idx_list[i]];
		node_cnt += g.num_nodes;
		prefix_sum.push_back(g.num_nodes);
		if (i > st)
			prefix_sum[i - st] += prefix_sum[i - st - 1];
	}
	for (int i = (int)prefix_sum.size() - 1; i > 0; --i)
		prefix_sum[i] = prefix_sum[i - 1]; // shift
	prefix_sum[0] = 0;	
	
	graph.Resize(num, node_cnt);
    
	x_cpu.Zeros(node_cnt, cfg::node_dim);
	y_cpu.Resize(num, cfg::num_class);
	y_cpu.ResizeSp(num, num + 1); 
	
	// labeling nodes, parsing node features
	Dtype* ptr = x_cpu.data;
	for (unsigned i = st; i < ed; ++i)
	{
		auto& g = graph_data[idx_list[i]];
		for (int j = 0; j < g.num_nodes; ++j)
		{
			ptr[g.node_label[j]] = 1.0;
			ptr += cfg::node_dim;
			graph.AddNode(i - st, prefix_sum[i - st] + j);
		}
	}
	
	// add edges, parsing edge features
	int x, y, cur_edge = 0;
	for (unsigned i = st; i < ed; ++i)
	{
		auto& g = graph_data[idx_list[i]];	
		for (int j = 0; j < g.num_nodes; ++j)
		{
			x = prefix_sum[i - st] + j;
			for (size_t k = 0; k < g.adj.head[j].size(); ++k)
			{
				y = prefix_sum[i - st] + g.adj.head[j][k];
				graph.AddEdge(cur_edge, x, y);	
				cur_edge++;
			}
		}
	}
	
    for (unsigned i = st; i < ed; ++i)
    {
    	assert(labels[ idx_list[i] ] >= 0 && labels[ idx_list[i] ] < cfg::num_class);
        y_cpu.data->ptr[i - st] = i - st;
        y_cpu.data->val[i - st] = 1.0;
        y_cpu.data->col_idx[i - st] = labels[ idx_list[i] ];  
    }
    y_cpu.data->ptr[num] = num;

	input.CopyFrom(x_cpu);
	label.CopyFrom(y_cpu);    
}

void MainLoop()
{
    learner = new MomentumSGDLearner<mode, Dtype>(&model, cfg::lr, cfg::momentum, cfg::l2_penalty);
    
	int max_iter = (long long)cfg::max_epoch * (long long)train_idx.size() / cfg::batch_size;
	unsigned cur_pos = 0;
	int init_iter = cfg::iter;
	if (init_iter > 0)
	{
		std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
		//gnn.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, init_iter));
	}
	
	Dtype nll, err;
	for (; cfg::iter <= max_iter; ++cfg::iter, cur_pos += cfg::batch_size)
	{
		if (cfg::iter % cfg::test_interval == 0)
		{			
			std::cerr << "testing" << std::endl;
			nll = err = 0.0;
			for (unsigned i = 0; i < test_idx.size(); i += cfg::batch_size)
			{
				GetBatch(test_idx, i, cfg::batch_size);
                model.SetupConstParams(init_const_dict); 
				gnn.FeedForward({{"data", &input}, {"label", &label}}, TEST);
				auto loss_map = gnn.GetLoss();
				nll += loss_map["classnll"];
			 	err += loss_map["errcnt"];
			}
			nll /= test_idx.size();
			err /= test_idx.size();
			std::cerr << fmt::sprintf("test nll: %.4f\t test err: %.4f", nll, err) << std::endl;			
		}
		
		if (cfg::iter % cfg::save_interval == 0 && cfg::iter != init_iter)
		{			
			printf("saving model for iter=%d\n", cfg::iter);			
			//gnn.Save(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, cfg::iter));
		}
		
		if (cur_pos + cfg::batch_size > train_idx.size())
		{
			std::random_shuffle(train_idx.begin(), train_idx.end());
			cur_pos = 0;
		}
	
		GetBatch(train_idx, cur_pos, cfg::batch_size);
        model.SetupConstParams(init_const_dict); 
		gnn.FeedForward({{"data", &input}, {"label", &label}}, TRAIN);
		auto loss_map = gnn.GetLoss();
		
    	if (cfg::iter % cfg::report_interval == 0)
		{
			std::cerr << fmt::sprintf("train iter=%d\tnll: %.4f\terr: %.4f", cfg::iter, loss_map["classnll"] / cfg::batch_size, loss_map["errcnt"] / cfg::batch_size) << std::endl;
		}
		
		gnn.BackPropagation();
        learner->Update();
	}
}

#endif
