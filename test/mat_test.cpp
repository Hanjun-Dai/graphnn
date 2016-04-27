#include "dense_matrix.h"

typedef double Dtype;
const MatMode mode = MatMode::GPU;

int main()
{
    GPUHandle::Init(0);
    DenseMat<mode, Dtype> a(3, 5);
    
    a.SetRandN(0, 1);
    a.Print2Screen();
    
    DenseMat<mode, Dtype> d;
    
    d.GetColsFrom(a, 1, 2);
    d.Print2Screen();
    
    GPUHandle::Destroy();
    return 0;
}