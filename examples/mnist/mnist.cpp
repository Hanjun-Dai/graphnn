#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include "mnist_helper.h"
#include "nn/param_set.h"
#include "tensor/sparse_tensor.h"
#include "nn/expr_sugar.h"
#include "nn/factor_graph.h"
#include "nn/matmul.h"
#include "nn/relu.h"
#include "nn/cross_entropy.h"
#include "nn/arg_max.h"
#include "nn/type_cast.h"
#include "nn/reduce_mean.h"
#include "nn/in_top_k.h"


using namespace gnn;

const char* f_train_feat, *f_train_label, *f_test_feat, *f_test_label;
unsigned batch_size = 100;
int dev_id;
std::vector< Dtype* > images_train, images_test;
std::vector< int > labels_train, labels_test;
typedef CPU mode;

void LoadParams(const int argc, const char** argv)
{
	for (int i = 1; i < argc; i += 2)
	{
		if (strcmp(argv[i], "-train_feat") == 0)
			f_train_feat = argv[i + 1];
        if (strcmp(argv[i], "-train_label") == 0)
			f_train_label = argv[i + 1];
        if (strcmp(argv[i], "-test_feat") == 0)
			f_test_feat = argv[i + 1];
        if (strcmp(argv[i], "-test_label") == 0)
			f_test_label = argv[i + 1];
        if (strcmp(argv[i], "-device") == 0)
			dev_id = atoi(argv[i + 1]);                                                                
	}
}

ParamSet<mode, Dtype> pset;
FactorGraph g;

std::pair<std::shared_ptr< DTensorVar<mode, Dtype> >, std::shared_ptr< DTensorVar<mode, Dtype> > > BuildGraph()
{
	auto w1 = add_diff<DTensorVar>(pset, "w1", {784, 1024});	
	auto w2 = add_diff<DTensorVar>(pset, "w2", {1024, 1024});
	auto wo = add_diff<DTensorVar>(pset, "wo", {1024, 10});
	w1->value.SetRandN(0, 0.01);
	w2->value.SetRandN(0, 0.01);
	wo->value.SetRandN(0, 0.01);

    g.AddVar(w1, false);
    g.AddVar(w2, false);
    g.AddVar(wo, false);

	auto x = add_var< DTensorVar<mode, Dtype> >(g, "x");
	auto y = add_var< SpTensorVar<mode, Dtype> >(g, "y");
	auto h1 = af< MatMul >({x, w1});

	h1 = af< ReLU >({h1});
	auto h2 = af< MatMul >({h1, w2});	
	h2 = af< ReLU >({h2});
	auto output = af< MatMul >({h2, wo});

	auto ce = af< CrossEntropy >({output, y}, true);
	auto loss = af< ReduceMean >({ce});

    auto label = af< ArgMax >({y});    

    // output (Dtype) and label (int) has different types
    // c++ doesn't have heterogeneous initializer list, make_pair is required
    // return an int tensor (vector)
    auto cmp = af< InTopK<mode, Dtype> >(std::make_pair(output, label));

	auto acc = af< ReduceMean >({ af< TypeCast<mode, Dtype> >({cmp}) });

	return {loss, acc};	
}

DTensor<CPU, Dtype> x_cpu;
SpTensor<CPU, Dtype> y_cpu;
DTensor<mode, Dtype> input;
SpTensor<mode, Dtype> label;

void LoadBatch(unsigned idx_st, std::vector< Dtype* >& images, std::vector< int >& labels)
{
    unsigned cur_bsize = batch_size;
    if (idx_st + batch_size > images.size())
        cur_bsize = images.size() - idx_st;
    x_cpu.Reshape({cur_bsize, 784});
    y_cpu.Reshape({cur_bsize, 10});
    y_cpu.ResizeSp(cur_bsize, cur_bsize + 1); 
    for (unsigned i = 0; i < cur_bsize; ++i)
    {
        memcpy(x_cpu.data->ptr + i * 784, images[i + idx_st], sizeof(Dtype) * 784); 
        y_cpu.data->row_ptr[i] = i;
        y_cpu.data->val[i] = 1.0;
        y_cpu.data->col_idx[i] = labels[i + idx_st];  
    }
    y_cpu.data->row_ptr[cur_bsize] = cur_bsize;

    input.CopyFrom(x_cpu);
    label.CopyFrom(y_cpu);
}

int main(const int argc, const char** argv)
{
	LoadParams(argc, argv); 
    LoadRaw(f_train_feat, f_train_label, images_train, labels_train);
    LoadRaw(f_test_feat, f_test_label, images_test, labels_test);
    std::cerr << images_train.size() << " images for training" << std::endl;
    std::cerr << images_test.size() << " images for test" << std::endl;

    auto targets = BuildGraph();
    auto var_loss = targets.first;
    auto var_acc = targets.second;

	Dtype loss, err_rate;  
	for (int epoch = 0; epoch < 10; ++epoch)
    {
        std::cerr << "testing" << std::endl;
        loss = err_rate = 0;
        for (unsigned i = 0; i < labels_test.size(); i += batch_size)
        {
                LoadBatch(i, images_test, labels_test);
                g.FeedForward({var_loss, var_acc}, {{"x", &input}, {"y", &label}});
                loss += var_loss->AsScalar() * input.rows();
                err_rate += (1.0 - var_acc->AsScalar()) * input.rows();
        }
        loss /= labels_test.size();
        err_rate /= labels_test.size();
        std::cerr << fmt::sprintf("test loss: %.4f\t error rate: %.4f", loss, err_rate) << std::endl;
        break;

        for (unsigned i = 0; i < labels_train.size(); i += batch_size)
        {
                LoadBatch(i, images_train, labels_train);
                g.FeedForward({var_loss, var_acc}, {{"x", &input}, {"y", &label}});
                g.BackPropagate({var_loss});
        }
    }
	return 0;
}