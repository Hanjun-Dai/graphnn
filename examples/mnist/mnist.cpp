#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include "mnist_helper.h"
#include "tensor/tensor_all.h"
#include "nn/nn_all.h"
#include "util/fmt.h"

using namespace gnn;

const char* f_train_feat, *f_train_label, *f_test_feat, *f_test_label;
unsigned batch_size = 100;
Dtype lr = 0.001;
int dev_id;
std::vector< Dtype* > images_train, images_test;
std::vector< int > labels_train, labels_test;

#ifdef USE_GPU
typedef GPU mode;
#else
typedef CPU mode;
#endif

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

std::shared_ptr< DTensorVar<mode, Dtype> > prob;

std::pair<std::shared_ptr< DTensorVar<mode, Dtype> >, std::shared_ptr< DTensorVar<mode, Dtype> > > BuildGraph()
{
    auto w1 = add_diff<DTensorVar>(pset, "w1", {785, 1024});    
    auto w2 = add_diff<DTensorVar>(pset, "w2", {1025, 1024});
    auto wo = add_diff<DTensorVar>(pset, "wo", {1025, 10});
    w1->value.SetRandN(0, 0.01);
    w2->value.SetRandN(0, 0.01);
    wo->value.SetRandN(0, 0.01);

    g.AddParam(w1);
    g.AddParam(w2);
    g.AddParam(wo);

    auto x = add_const< DTensorVar<mode, Dtype> >(g, "x", true);
    auto y = add_const< SpTensorVar<mode, Dtype> >(g, "y", true);
    auto h1 = af< FullyConnected >(g, {x, w1});

    h1 = af< ReLU >(g, {h1});
    auto h2 = af< FullyConnected >(g, {h1, w2});    
    h2 = af< ReLU >(g, {h2});
    auto output = af< FullyConnected >(g, {h2, wo});
        prob = af< Softmax >(g, {output});
    auto ce = af< CrossEntropy >(g, {prob, y}, false);
    auto loss = af< ReduceMean >(g, {ce});

    auto label = af< ArgMax >(g, {y});    

    // output (Dtype) and label (int) has different types
    // c++ doesn't have heterogeneous initializer list, make_pair is required
    // return an int tensor (vector)
    auto cmp = af< InTopK<mode, Dtype> >(g, std::make_pair(output, label));
    auto real_cmp = af< TypeCast<mode, Dtype> >(g, {cmp});

    auto acc = af< ReduceMean >(g, { real_cmp });

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
    GpuHandle::Init(dev_id, 1);

    LoadRaw(f_train_feat, f_train_label, images_train, labels_train);
    LoadRaw(f_test_feat, f_test_label, images_test, labels_test);
    std::cerr << images_train.size() << " images for training" << std::endl;
    std::cerr << images_test.size() << " images for test" << std::endl;

    auto targets = BuildGraph();
    auto var_loss = targets.first;
    auto var_acc = targets.second;

    MomentumSGDOptimizer<mode, Dtype> optmz(&pset, lr, 0.9, 0);
    //AdamOptimizer<mode, Dtype> optmz(&pset, lr);
    optmz.clipping_enabled = false;

    Dtype loss, err_rate;
    
    for (int epoch = 0; epoch < 10; ++epoch)
    {
        pset.Save(fmt::sprintf("epoch_%d.model", epoch));
        std::cerr << "testing" << std::endl;
        loss = err_rate = 0;
        for (unsigned i = 0; i < labels_test.size(); i += batch_size)
        {
                LoadBatch(i, images_test, labels_test);
                g.FeedForward({var_loss, var_acc}, {{"x", &input}, {"y", &label}}, Phase::TEST);
                // output predicted probabilities
                /* 
                DTensor<CPU, Dtype> prob_cpu;
                prob_cpu.CopyFrom(prob->value);
                for (size_t j = 0; j < prob_cpu.rows(); ++j)
                {
                    for (size_t k = 0; k < prob_cpu.cols(); ++k)
                        printf("%.2f ", prob_cpu.data->ptr[j * prob_cpu.cols() + k]);
                    printf("\n");
                }
                */
                loss += var_loss->AsScalar() * input.rows();
                err_rate += (1.0 - var_acc->AsScalar()) * input.rows();
        }
        loss /= labels_test.size();
        err_rate /= labels_test.size();
        std::cerr << fmt::sprintf("test loss: %.4f\t error rate: %.4f", loss, err_rate) << std::endl;

        for (unsigned i = 0; i < labels_train.size(); i += batch_size)
        {
                LoadBatch(i, images_train, labels_train);
                g.FeedForward({var_loss, var_acc}, {{"x", &input}, {"y", &label}}, Phase::TRAIN);

                g.BackPropagate({var_loss});

                optmz.Update();
        }
    }
    GpuHandle::Destroy();
    return 0;
}
