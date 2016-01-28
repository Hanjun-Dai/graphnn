#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include "dense_matrix.h"
#include "graphnn.h"
#include "param_layer.h"
#include "input_layer.h"
#include "ada_fastfood_param.h"
#include "cppformat/format.h"
#include "relu_layer.h"
#include "nonshared_linear_param.h"
#include "classnll_criterion_layer.h"
#include "err_cnt_criterion_layer.h"
#include "batch_norm_param.h"

typedef double Dtype;
const MatMode mode = GPU;
const char* f_train_feat, *f_train_label, *f_test_feat, *f_test_label;
unsigned batch_size = 100;
int dev_id;
Dtype lr = 0.001;
unsigned dim = 784;

GraphData<mode, Dtype> g_input(DENSE), g_label(SPARSE);
GraphNN<mode, Dtype> gnn;
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
    auto* input_layer = new InputLayer<mode, Dtype>("input", GraphAtt::NODE);
    
    auto* h1_weight = new LinearParam<mode, Dtype>("h1_weight", dim, 1024, 0, 0.01);
	auto* h1 = new SingleParamNodeLayer<mode, Dtype>("h1", h1_weight, GraphAtt::NODE);
    /*
    auto* bn_1 = new BatchNormParam<mode, Dtype>("bn1", GraphAtt::NODE, 1024, true);
    auto* bn_layer_1 = new NodeLayer<mode, Dtype>("bn_layer_1");
    bn_layer_1->AddParam(h1->name, bn_1);
    */
    auto* relu_1 = new ReLULayer<mode, Dtype>("relu_1", GraphAtt::NODE, WriteType::INPLACE);
    
    auto* h2_weight = new LinearParam<mode, Dtype>("h2_weight", 1024, 1024, 0, 0.01);
	auto* h2 = new SingleParamNodeLayer<mode, Dtype>("h2", h2_weight, GraphAtt::NODE);
    /*
    auto* bn_2 = new BatchNormParam<mode, Dtype>("bn2", GraphAtt::NODE, 1024, true);
    auto* bn_layer_2 = new NodeLayer<mode, Dtype>("bn_layer_2");
    bn_layer_2->AddParam(h2->name, bn_2);
    */
    auto* relu_2 = new ReLULayer<mode, Dtype>("relu_2", GraphAtt::NODE, WriteType::INPLACE);
    
    auto* o_weight = new LinearParam<mode, Dtype>("o_weight", 1024, 10, 0, 0.01);
	auto* output = new SingleParamNodeLayer<mode, Dtype>("output", o_weight, GraphAtt::NODE);
    
    auto* classnll = new ClassNLLCriterionLayer<mode, Dtype>("classnll", true);
    auto* errcnt = new ErrCntCriterionLayer<mode, Dtype>("errcnt");
    
    gnn.AddParam(h1_weight); 
	gnn.AddParam(h2_weight);
    gnn.AddParam(o_weight);
    //gnn.AddParam(bn_1); 
    //gnn.AddParam(bn_2); 
    
    gnn.AddLayer(input_layer);
	gnn.AddLayer(h1);
    //gnn.AddLayer(bn_layer_1);
	gnn.AddLayer(relu_1);
    gnn.AddLayer(h2);
    //gnn.AddLayer(bn_layer_2);
    gnn.AddLayer(relu_2);
	gnn.AddLayer(output);
    
    gnn.AddEdge(input_layer, h1);
	gnn.AddEdge(h1, relu_1);
    //gnn.AddEdge(bn_layer_1, relu_1); 
	gnn.AddEdge(relu_1, h2);
    gnn.AddEdge(h2, relu_2);
    //gnn.AddEdge(bn_layer_2, relu_2);
    gnn.AddEdge(relu_2, output); 
           
    gnn.AddEdge(output, classnll);    
    gnn.AddEdge(output, errcnt);    
}

void LoadBatch(unsigned idx_st, std::vector< Dtype* >& images, std::vector< int >& labels)
{
    g_input.graph->Resize(1, batch_size);
	g_label.graph->Resize(1, batch_size);
    
    x_cpu.Resize(batch_size, 784);
    y_cpu.Resize(batch_size, 10);
    y_cpu.ResizeSp(batch_size, batch_size + 1); 
    
    for (unsigned i = 0; i < batch_size; ++i)
    {
        memcpy(x_cpu.data + i * 784, images[i + idx_st], sizeof(Dtype) * 784); 
        y_cpu.data->ptr[i] = i;
        y_cpu.data->val[i] = 1.0;
        y_cpu.data->col_idx[i] = labels[i + idx_st];  
    }
    y_cpu.data->ptr[batch_size] = batch_size;
    
    g_input.node_states->DenseDerived().CopyFrom(x_cpu);
    g_label.node_states->SparseDerived().CopyFrom(y_cpu);
}

int main(const int argc, const char** argv)
{	
    LoadParams(argc, argv);    
	GPUHandle::Init(dev_id);
    InitModel();
    LoadRaw(f_train_feat, f_train_label, images_train, labels_train);
    LoadRaw(f_test_feat, f_test_label, images_test, labels_test);
        
    Dtype loss, err_rate;       
    for (int epoch = 0; epoch < 10; ++epoch)
    {
        std::cerr << "testing" << std::endl;
        loss = err_rate = 0;
        for (unsigned i = 0; i < labels_test.size(); i += batch_size)
        {
                LoadBatch(i, images_test, labels_test);
        		gnn.ForwardData({{"input", &g_input}}, TEST);                               								
				auto loss_map = gnn.ForwardLabel({{"classnll", &g_label},
                                                  {"errcnt", &g_label}});                
				loss += loss_map["classnll"];
                err_rate += loss_map["errcnt"];
        }
        loss /= labels_test.size();
        err_rate /= labels_test.size();
        std::cerr << fmt::sprintf("test loss: %.4f\t error rate: %.4f", loss, err_rate) << std::endl;
        
        for (unsigned i = 0; i < labels_train.size(); i += batch_size)
        {
                LoadBatch(i, images_train, labels_train);
                gnn.ForwardData({{"input", &g_input}}, TRAIN);
                auto loss_map = gnn.ForwardLabel({{"classnll", &g_label}});
				loss = loss_map["classnll"] / batch_size;
                
                gnn.BackPropagation();
		        gnn.UpdateParams(lr, 0, 0);                                             
        }
    }            
    
    GPUHandle::Destroy();
	return 0;    
}	
