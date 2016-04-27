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
#include "max_entropy_criterion_layer.h"
#include "model.h"
#include "learner.h"
#include "concat_layer.h"

typedef double Dtype;
const MatMode mode = CPU;

DenseMat<mode, Dtype> input;
NNGraph<mode, Dtype> g;
Model<mode, Dtype> model;

int main()
{
    GPUHandle::Init(0);
    input.Resize(1, 10);
    input.SetRandN(0, 10);
    
    auto* h1_weight = add_diff< LinearParam >(model, "w_h1", input.cols, 4, 0, 0.01);
    auto* h2_weight = add_diff< LinearParam >(model, "w_h2", input.cols, 4, 0, 0.01);    
    auto* data_layer = cl< InputLayer >("data", g, {});
    
    auto* h1 = cl< ParamLayer >(g, {data_layer}, {h1_weight});
    auto* h2 = cl< ParamLayer >(g, {data_layer}, {h2_weight});
    auto* h = cl< ConcatLayer >(g, {h1, h2});
    
    auto* p = cl< SoftmaxLayer >("p", g, {h});            
    cl< MaxEntropyCriterionLayer >("entropy", g, {p});
    
    MomentumSGDLearner<mode, Dtype> learner(&model, 0.001, 0.9, 0);
    
    DenseMat<mode, Dtype> buf;
    for (int i = 0; i < 1000; ++i)
    {
        g.FeedForward({{"data", &input}}, TRAIN);
        
        g.GetState("p", buf);
        buf.Print2Screen();
        
        auto loss_map = g.GetLoss();
        
        if (i % 1 == 0)
            std::cerr << "iter: " << i << " loss: " << loss_map["entropy"] << std::endl;
            
        g.BackPropagation();
        learner.Update();            
    }        
    
    GPUHandle::Destroy();    
    return 0;
}