#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "dense_matrix.h"
#include "sparse_matrix.h"
#include "cppformat/format.h"
#include "nngraph_expr.h"

typedef double Dtype;
const MatMode mode = GPU;
using namespace std;


int main()
{	
	GPUHandle::Init(0);

    NNGraphExpr<mode, Dtype> input(std::make_shared< InputLayer<mode, Dtype> >());
    NNGraphExpr<mode, Dtype> input2(std::make_shared< InputLayer<mode, Dtype> >());
    
    auto expr = input + input2;
    
    
	GPUHandle::Destroy();
	return 0;
}
