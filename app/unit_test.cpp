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
#include "node_layer.h"
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
	        
    DenseMat<mode, Dtype> a(1, 5);
    
    a.Fill(14);    
    a.Print2Screen();
    cudaMemset(a.data + 1, 0, 3 * sizeof(Dtype));    
    a.Print2Screen();
    
	GPUHandle::Destroy();
	return 0;
}
