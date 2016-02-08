#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "linear_param.h"
#include "graphnn.h"
#include "param_layer.h"
#include "input_layer.h"
#include "cppformat/format.h"
#include "mse_criterion_layer.h"
#include "fast_wht.h"
#include "ada_fastfood_param.h"
#include "sin_layer.h"
#include "cos_layer.h"
#include "batch_norm_param.h"
#include "softmax_layer.h"
#include "const_trans_layer.h"
#include "elewise_mul_layer.h"

typedef double Dtype;
const MatMode mode = CPU;
using namespace std;

int main()
{	
	GPUHandle::Init(0);
    
    auto* layer = new PairMulLayer<mode, Dtype>("mul", GraphAtt::NODE);
    
    auto* prev_1 = new ConstTransLayer<mode, Dtype>("a", GraphAtt::NODE, 1, 1);
    auto* prev_2 = new ConstTransLayer<mode, Dtype>("b", GraphAtt::NODE, 1, 1);
    
    DenseMat<CPU, Dtype> a(3, 2), b(3, 2);
    a.Fill(-1.0);
    b.Fill(3.0);
    
    GraphData<mode, Dtype> g1(DENSE), g2(DENSE);
    
    g1.node_states->DenseDerived().CopyFrom(a);
    g2.node_states->DenseDerived().CopyFrom(b);
    
    prev_1->graph_output = &g1;
    prev_2->graph_output = &g2;
    
    layer->UpdateOutput(prev_1, SvType::WRITE2, Phase::TRAIN);
    layer->UpdateOutput(prev_2, SvType::ADD2, Phase::TRAIN);
    
    auto& output = layer->graph_output->node_states->DenseDerived();
    output.Print2Screen();
    
    auto& cur_grad = GetImatState(layer->graph_gradoutput, layer->at)->DenseDerived();
    cur_grad.SetRandN(0, 0.1, 3, 2);
    cur_grad.Print2Screen();
    layer->BackPropErr(prev_1, SvType::WRITE2);
    layer->BackPropErr(prev_2, SvType::WRITE2);
    
    prev_1->graph_gradoutput->node_states->DenseDerived().Print2Screen();
    prev_2->graph_gradoutput->node_states->DenseDerived().Print2Screen();
    
	GPUHandle::Destroy();
	return 0;
}
