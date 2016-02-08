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

typedef float Dtype;
using namespace std;


template<MatMode mode>
void check(DenseMat<CPU, Dtype>& input, SparseMat<CPU, Dtype>& mask)
{
    DenseMat<mode, Dtype> a;
    SparseMat<mode, Dtype> c;
    
    a.CopyFrom(input);
    c.CopyFrom(mask);
 
    a.EleWiseMul(c);   
    a.Print2Screen();
}

int main()
{	
	GPUHandle::Init(0);
	        
    DenseMat<CPU, Dtype> input(3, 5);
    SparseMat<CPU, Dtype> mask(3, 5);
    mask.ResizeSp(4, 4);
    
    mask.data->ptr[0] = 0; mask.data->ptr[1] = 2; mask.data->ptr[2] = 3; mask.data->ptr[3] = 4;
    mask.data->col_idx[0] = 0; mask.data->col_idx[1] = 2; mask.data->col_idx[2] = 3; mask.data->col_idx[3] = 4;
    for (int i = 0; i < 4; ++i)
        mask.data->val[i] = 2.0;
        
    input.Fill(1.0);

                
    check<CPU>(input, mask);
    //check<GPU>(input, deriv);
    
	GPUHandle::Destroy();
	return 0;
}
