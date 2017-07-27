#include "nn/jagged_softmax.h"
#include "tensor/mkl_helper.h"

namespace gnn
{

template<typename Dtype>
void JaggedSoftmaxDeriv(DTensor<CPU, Dtype>& dst, DTensor<CPU, Dtype>& cur_output, DTensor<CPU, Dtype>& cur_grad, DTensor<CPU, int>& lens)
{
	DTensor<CPU, Dtype> buf(cur_grad.shape);
    buf.CopyFrom(cur_grad);
    
    Dtype z;
    size_t offset = 0;
    int n_c = 0;
    for (size_t i = 0; i < lens.shape.Count(); ++i)
    {
    	n_c = lens.data->ptr[i];
    	z = MKL_Dot(n_c, cur_grad.data->ptr + offset, cur_output.data->ptr + offset);
    	for (int j = 0; j < n_c; ++j)
            buf.data->ptr[offset + j] -= z;

        offset += n_c;
    }
    ASSERT( offset == dst.shape.Count(), "length mismatch in jagged softmax");
    
    buf.ElewiseMul(cur_output);
    
    dst.Axpy(1.0, buf);
}

template<typename mode, typename Dtype>
JaggedSoftmax<mode, Dtype>::JaggedSoftmax(std::string _name, PropErr _properr) 
					: Factor(_name, _properr)
{
}

template<typename mode, typename Dtype>
void JaggedSoftmax<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 		std::vector< std::shared_ptr<Variable> >& outputs, 
						 		Phase phase) 
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType());

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& lens = dynamic_cast<DTensorVar<mode, int>*>(operands[1].get())->value;

	output.CopyFrom(input);

	output.JaggedSoftmax(lens);
}

template<typename mode, typename Dtype>
void JaggedSoftmax<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
								std::vector< bool >& isConst, 
						 		std::vector< std::shared_ptr<Variable> >& outputs) 
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 
	if (isConst[0])
		return;
	auto& lens = dynamic_cast<DTensorVar<mode, int>*>(operands[1].get())->value;

	auto* var_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get());
	auto& cur_out = var_out->value;
	auto cur_grad = var_out->grad.Full();

	auto prev_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();

	JaggedSoftmaxDeriv(prev_grad, cur_out, cur_grad, lens);
}

INSTANTIATE_CLASS(JaggedSoftmax)

}