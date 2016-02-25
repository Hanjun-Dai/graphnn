#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "cppformat/format.h"
#include "nngraph.h"
#include "i_layer.h"
#include "linear_param.h"
#include "learner.h"
#include "model.h"
#include "param_layer.h"

typedef double Dtype;
const MatMode mode = GPU;
using namespace std;

int main()
{	
	GPUHandle::Init(0);

    NNGraph<mode, Dtype> nn;
        
    IParam<mode, Dtype>* param = new IParam<mode, Dtype>();
    ILayer<mode, Dtype>* lp, *lq;
    param->OutSize();
    auto* l1 = nn.cl< ParamLayer >("a", {lp, lq}, param);

	GPUHandle::Destroy();
	return 0;
}
