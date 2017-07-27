#include "nn/softmax.h"
#include "tensor/mkl_helper.h"

namespace gnn
{

template<typename Dtype>
void SoftmaxDeriv(DTensor<CPU, Dtype>& dst, DTensor<CPU, Dtype>& cur_output, DTensor<CPU, Dtype>& cur_grad)
{
	DTensor<CPU, Dtype> buf(cur_grad.shape);
    buf.CopyFrom(cur_grad);
    
    Dtype z;    
    size_t offset = 0;
    auto n_row = buf.rows(), n_col = buf.cols();
    for (size_t i = 0; i < n_row; ++i)
    {
        z = MKL_Dot(n_col, cur_grad.data->ptr + offset, cur_output.data->ptr + offset);
        
        for (size_t j = 0; j < n_col; ++j)
            buf.data->ptr[offset + j] -= z;
        
        offset += n_col; 
    }
    
    buf.ElewiseMul(cur_output);
    
    dst.Axpy(1.0, buf);
}

template<typename mode, typename Dtype>
Softmax<mode, Dtype>::Softmax(std::string _name, PropErr _properr) 
					: Factor(_name, _properr)
{
}

template<typename mode, typename Dtype>
void Softmax<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 		std::vector< std::shared_ptr<Variable> >& outputs, 
						 		Phase phase) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	output.CopyFrom(input);
	output.Softmax();
}

template<typename mode, typename Dtype>
void Softmax<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
								std::vector< bool >& isConst, 
						 		std::vector< std::shared_ptr<Variable> >& outputs) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 
	if (isConst[0])
		return;
	auto* var_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get());
	auto& cur_out = var_out->value;
	auto cur_grad = var_out->grad.Full();

	auto prev_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();

	SoftmaxDeriv(prev_grad, cur_out, cur_grad);
}

INSTANTIATE_CLASS(Softmax)

}