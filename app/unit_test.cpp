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

typedef double Dtype;
const MatMode mode = GPU;
using namespace std;

int main()
{	
	GPUHandle::Init(0);
	        
    DenseMat<mode, Dtype> a(100, 100);
    
    a.SetRandN(0, 0.01);
    std::cerr << a.Norm2() << std::endl;
    
	GPUHandle::Destroy();
	return 0;
}
