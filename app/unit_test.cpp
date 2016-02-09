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
#include "expnll_criterion_layer.h"

typedef double Dtype;
const MatMode mode = GPU;
using namespace std;

int main()
{	
	GPUHandle::Init(0);
    
    DenseMat<mode, Dtype> a(4, 1), b(4, 1);
    
    a.Fill(2.0);
    std::cerr << a.Norm2() << std::endl;
    b.Fill(3.0);
    GraphData<mode, Dtype> g(DENSE), y(DENSE);
    g.node_states->DenseDerived().CopyFrom(a);
    y.node_states->DenseDerived().CopyFrom(b);
    
    auto* layer = new MSECriterionLayer<mode, Dtype>("exp", 2.0);
    
    layer->graph_output = &g;
    
    std::cerr << layer->GetLoss(&y) << std::endl;
    
    auto& grad = layer->graph_gradoutput->node_states->DenseDerived();
    grad.Print2Screen();
    
	GPUHandle::Destroy();
	return 0;
}
