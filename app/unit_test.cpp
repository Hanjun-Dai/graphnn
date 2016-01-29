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

typedef double Dtype;
using namespace std;

template<MatMode mode>
void check(DenseMat<CPU, Dtype>& input, DenseMat<CPU, Dtype>& deriv)
{
    DenseMat<mode, Dtype> a, c;
    a.CopyFrom(input);
    c.CopyFrom(deriv);
        
    a.Softmax();
        
    a.Print2Screen();
    
    auto* l = new SoftmaxLayer<mode, Dtype>("softmax", GraphAtt::NODE, WriteType::INPLACE);
              
    DenseMat<mode, Dtype> b(3, 5);
    
    l->Derivative(b, a, a, c);
     
    b.Print2Screen();    
}

int main()
{	
	GPUHandle::Init(0);
	        
    DenseMat<CPU, Dtype> input(3, 5), deriv(3, 5);
    input.SetRandN(0, 1);
    deriv.SetRandN(0, 1);
                
    check<CPU>(input, deriv);
    check<GPU>(input, deriv);
    
	GPUHandle::Destroy();
	return 0;
}
