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
#include "ilayer.h"
#include "iparam.h"
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
    
    param->OutSize();
    auto* l1 = nn.cl< ParamLayer >({nullptr}, param);

	GPUHandle::Destroy();
	return 0;
}
