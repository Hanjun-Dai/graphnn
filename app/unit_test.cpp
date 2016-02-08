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
const MatMode mode = GPU;
using namespace std;

int main()
{	
	GPUHandle::Init(0);
    
    
	GPUHandle::Destroy();
	return 0;
}
