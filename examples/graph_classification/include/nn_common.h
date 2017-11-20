#ifndef NN_COMMON_H
#define NN_COMMON_H

#include <random>
#include <algorithm>

#include "config.h"
#include "utils.h"
#include "util/fmt.h"
#include "tensor/tensor_all.h"
#include "nn/nn_all.h"

using namespace gnn;

typedef GPU mode;

std::vector< Graph > graph_data;
std::vector<int> labels;
std::vector<int> train_idx, test_idx;

FactorGraph fg;
ParamSet<mode, Dtype> model;
MomentumSGDOptimizer<mode, Dtype>* learner;
std::map< std::string, void* > inputs;
std::vector< std::shared_ptr< Variable > > targets;
std::vector< std::shared_ptr< Variable > > objs;

DTensor<CPU, Dtype> x_cpu;
SpTensor<CPU, Dtype> y_cpu;
GraphStruct graph;

DTensor<mode, Dtype> input;
SpTensor<mode, Dtype> label;

std::vector< unsigned > prefix_sum;
inline unsigned GetBatch(const std::vector<int>& idx_list, unsigned st, unsigned num)
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
    
    x_cpu.Reshape({node_cnt, (uint)cfg::node_dim});
	x_cpu.Zeros();
	y_cpu.Reshape({num, (uint)cfg::num_class});
	y_cpu.ResizeSp(num, num + 1);
	
	// labeling nodes, parsing node features
	Dtype* ptr = x_cpu.data->ptr;
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
        y_cpu.data->row_ptr[i - st] = i - st;
        y_cpu.data->val[i - st] = 1.0;
        y_cpu.data->col_idx[i - st] = labels[ idx_list[i] ];  
    }
    y_cpu.data->row_ptr[num] = num;

	input.CopyFrom(x_cpu);
	label.CopyFrom(y_cpu);    

	return num;
}

void MainLoop()
{
	inputs.clear();
	inputs["data"] = &input;
	inputs["label"] = &label;
	inputs["graph"] = &graph;

    learner = new MomentumSGDOptimizer<mode, Dtype>(&model, cfg::lr, cfg::momentum, cfg::l2_penalty);
    
	int max_iter = (long long)cfg::max_epoch * (long long)train_idx.size() / cfg::batch_size;
	unsigned cur_pos = 0;
	int init_iter = cfg::iter;
	if (init_iter > 0)
	{
		std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
		//gnn.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, init_iter));
	}
	
	for (; cfg::iter <= max_iter; ++cfg::iter, cur_pos += cfg::batch_size)
	{
		if (cfg::iter % cfg::test_interval == 0)
		{			
			std::cerr << "testing" << std::endl;
			std::map<std::string, Dtype> loss_total;
			for (unsigned i = 0; i < test_idx.size(); i += cfg::batch_size)
			{
				auto cur_bsize = GetBatch(test_idx, i, cfg::batch_size);
				fg.FeedForward(targets, inputs, Phase::TEST);
				for (auto& t : targets)
				{
					if (!loss_total.count(t->name))
						loss_total[t->name] = 0.0;
					loss_total[t->name] += cur_bsize * dynamic_cast<TensorVar<mode, Dtype>*>(t.get())->AsScalar();
				}
			}
			std::cerr << "test@iter: " << cfg::iter;
			for (auto t : loss_total)
				std::cerr << "\t" << t.first << ": " << t.second / test_idx.size();
			std::cerr << std::endl;
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
		fg.FeedForward(targets, inputs, Phase::TRAIN);
		
    	if (cfg::iter % cfg::report_interval == 0)
		{
			std::cerr << "iter: " << cfg::iter;	
			for (auto t : targets)
				std::cerr << "\t" << t->name << ": " << dynamic_cast<TensorVar<mode, Dtype>*>(t.get())->AsScalar();
			std::cerr << std::endl;
		}
		fg.BackPropagate(objs);
		learner->Update();
	}
}

#endif
