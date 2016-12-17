#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include "mnist_helper.h"
#include "nn/param_set.h"
#include "nn/factor_graph.h"
#include "nn/matmul.h"

using namespace gnn;

const char* f_train_feat, *f_train_label, *f_test_feat, *f_test_label;
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

ParamSet pset;
FactorGraph g;
void BuildGraph()
{
	auto w1 = add_diff<DTensorVar<mode, Dtype> >(pset, "w1", {784u, 1024u});	
	auto w2 = add_diff<DTensorVar<mode, Dtype> >(pset, "w2", {1024u, 1024u});
	auto wo = add_diff<DTensorVar<mode, Dtype> >(pset, "wo", {1024u, 10u});
	w1->value.SetRandN(0, 0.01);
	w2->value.SetRandN(0, 0.01);
	wo->value.SetRandN(0, 0.01);

	auto x = add_var< DTensorVar<mode, Dtype> >(g, "x");
	/*auto h1 =  add_factor< MatMul >(g, {x});
	h1 = add_factor<ReLU>(g, {h1});

	auto h2 = add_factor<MatMul>(g, {h1});
	h2 = add_factor<ReLU>(g, {h2});
 
	auto output = add_factor<MatMul>()*/
}

int main(const int argc, const char** argv)
{
	LoadParams(argc, argv); 
    LoadRaw(f_train_feat, f_train_label, images_train, labels_train);
    LoadRaw(f_test_feat, f_test_label, images_test, labels_test);
    std::cerr << images_train.size() << " images for training" << std::endl;
    std::cerr << images_test.size() << " images for test" << std::endl;
	return 0;
}