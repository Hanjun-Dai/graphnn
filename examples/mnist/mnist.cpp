#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include "dense_matrix.h"
#include "nngraph.h"
#include "param_layer.h"
#include "linear_param.h"
#include "input_layer.h"
#include "softmax_layer.h"
#include "relu_layer.h"
#include "classnll_criterion_layer.h"
#include "err_cnt_criterion_layer.h"
#include "model.h"
#include "learner.h"

typedef float Dtype;
const MatMode mode = CPU;
const char* f_train_feat, *f_train_label, *f_test_feat, *f_test_label;
unsigned batch_size = 100;
int dev_id;
Dtype lr = 0.001;
unsigned dim = 784;

DenseMat<mode, Dtype> input;
SparseMat<mode, Dtype> label;
NNGraph<mode, Dtype> g;
Model<mode, Dtype> model;

DenseMat<CPU, Dtype> x_cpu;
SparseMat<CPU, Dtype> y_cpu;

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

std::vector< Dtype* > images_train, images_test;
std::vector< int > labels_train, labels_test;

void LoadRaw(const char* f_image, const char* f_label, std::vector< Dtype* >& images, std::vector< int >& labels)
{
    FILE* fid = fopen(f_image, "r");
    int buf;
    assert(fread(&buf, sizeof(int), 1, fid) == 1); // magic number
    int num;
    assert(fread(&num, sizeof(int), 1, fid) == 1); // num
    num = __builtin_bswap32(num); // the raw data is high endian    
    assert(fread(&buf, sizeof(int), 1, fid) == 1); // rows 
    assert(fread(&buf, sizeof(int), 1, fid) == 1); // cols
    images.clear();    
    unsigned char* buffer = new unsigned char[dim];
    for (int i = 0; i < num; ++i)
    {
        assert(fread(buffer, sizeof(unsigned char), dim, fid) == dim);
        Dtype* img = new Dtype[dim];
        for (unsigned j = 0; j < dim; ++j)
            img[j] = buffer[j];
        images.push_back(img);            
    }    
    delete[] buffer;
    fclose(fid);    
    
    fid = fopen(f_label, "r");
    assert(fread(&buf, sizeof(int), 1, fid) == 1); // magic number    
    assert(fread(&num, sizeof(int), 1, fid) == 1); // num
    num = __builtin_bswap32(num); // the raw data is high endian
    buffer = new unsigned char[num];
    assert(fread(buffer, sizeof(unsigned char), num, fid) == (unsigned)num);
    fclose(fid);
    labels.clear();
    for (int i = 0; i < num; ++i)
        labels.push_back(buffer[i]);    
    delete[] buffer;        
}

void InitModel()
{
    auto* h1_weight = add_diff< LinearParam >(model, "w_h1", dim, 1024, 0, 0.01);
    auto* h2_weight = add_diff< LinearParam >(model, "w_h2", 1024, 1024, 0, 0.01);
    auto* o_weight = add_diff< LinearParam >(model, "w_o", 1024, 10, 0, 0.01);
    
    auto* data_layer = cl< InputLayer >("data", g, {});
    auto* label_layer = cl< InputLayer >("label", g, {});

    auto* h1 = cl< ParamLayer >(g, {data_layer}, {h1_weight});    
    auto* relu_1 = cl< ReLULayer >(g, {h1});     
    auto* h2 = cl< ParamLayer >(g, {relu_1}, {h2_weight});
    auto* relu_2 = cl< ReLULayer >(g, {h2});
    auto* output = cl< ParamLayer >(g, {relu_2}, {o_weight});
    
    cl< ClassNLLCriterionLayer >("classnll", g, {output, label_layer}, true);
    cl< ErrCntCriterionLayer >("errcnt", g, {output, label_layer});
}

void LoadBatch(unsigned idx_st, std::vector< Dtype* >& images, std::vector< int >& labels)
{
    unsigned cur_bsize = batch_size;
    if (idx_st + batch_size > images.size())
        cur_bsize = images.size() - idx_st;
    x_cpu.Resize(cur_bsize, 784);
    y_cpu.Resize(cur_bsize, 10);
    y_cpu.ResizeSp(cur_bsize, cur_bsize + 1); 
    
    for (unsigned i = 0; i < cur_bsize; ++i)
    {
        memcpy(x_cpu.data + i * 784, images[i + idx_st], sizeof(Dtype) * 784); 
        y_cpu.data->ptr[i] = i;
        y_cpu.data->val[i] = 1.0;
        y_cpu.data->col_idx[i] = labels[i + idx_st];  
    }
    y_cpu.data->ptr[cur_bsize] = cur_bsize;
    
    input.CopyFrom(x_cpu);
    label.CopyFrom(y_cpu);
}

int main(const int argc, const char** argv)
{	
    LoadParams(argc, argv);    
//	GPUHandle::Init(dev_id);
    
    InitModel();
    
    LoadRaw(f_train_feat, f_train_label, images_train, labels_train);
    LoadRaw(f_test_feat, f_test_label, images_test, labels_test);
    
    //MomentumSGDLearner<mode, Dtype> learner(&model, lr, 0.9, 0);
    AdamLearner<mode, Dtype> learner(&model, lr);
    learner.clipping_enabled = false;

    Dtype loss, err_rate;       
    for (int epoch = 0; epoch < 10; ++epoch)
    {
        std::cerr << "testing" << std::endl;
        loss = err_rate = 0;
        for (unsigned i = 0; i < labels_test.size(); i += batch_size)
        {
                LoadBatch(i, images_test, labels_test);
        		g.FeedForward({{"data", &input}, {"label", &label}}, TEST);      								
				auto loss_map = g.GetLoss();

				loss += loss_map["classnll"];
                err_rate += loss_map["errcnt"];
        }
        loss /= labels_test.size();
        err_rate /= labels_test.size();
        std::cerr << fmt::sprintf("test loss: %.4f\t error rate: %.4f", loss, err_rate) << std::endl;
        
        for (unsigned i = 0; i < labels_train.size(); i += batch_size)
        {
                LoadBatch(i, images_train, labels_train);
                g.FeedForward({{"data", &input}, {"label", &label}}, TRAIN);
                
                g.BackPropagation();
                learner.Update();
        }
    }
    
//    GPUHandle::Destroy();
	return 0;    
}
