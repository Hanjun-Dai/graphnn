#include "nn/l2_col_norm.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_unary_functor.h"

namespace gnn
{

template<typename Dtype>
void L2ColNormFwd(DTensor<GPU, Dtype>& in, DTensor<GPU, Dtype>& out, DTensor<GPU, Dtype>& norm2, DTensor<GPU, Dtype>& len)
{
    
}

template void L2ColNormFwd(DTensor<GPU, float>& in, DTensor<GPU, float>& out, DTensor<GPU, float>& norm2, DTensor<GPU, float>& len);
template void L2ColNormFwd(DTensor<GPU, double>& in, DTensor<GPU, double>& out, DTensor<GPU, double>& norm2, DTensor<GPU, double>& len);

template<typename Dtype>
void L2ColNormGrad(DTensor<GPU, Dtype>& x, DTensor<GPU, Dtype>& prev_grad, DTensor<GPU, Dtype>& cur_grad, DTensor<GPU, Dtype>& norm2, DTensor<GPU, Dtype>& len, Dtype scale)
{

}

template void L2ColNormGrad(DTensor<GPU, float>& x, DTensor<GPU, float>& prev_grad, DTensor<GPU, float>& cur_grad, DTensor<GPU, float>& norm2, DTensor<GPU, float>& len, float scale);
template void L2ColNormGrad(DTensor<GPU, double>& x, DTensor<GPU, double>& prev_grad, DTensor<GPU, double>& cur_grad, DTensor<GPU, double>& norm2, DTensor<GPU, double>& len, double scale);


}
