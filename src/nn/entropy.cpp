#include "nn/entropy.h"
#include <cmath>

namespace gnn
{

template<typename Dtype>
void CalcEntropy(DTensor<CPU, Dtype>& prob, DTensor<CPU, Dtype>& out)
{
	ASSERT(prob.cols() == 1, "only support binary now");
	out.Reshape({prob.rows(), 1});
	for (size_t i = 0; i < prob.rows(); ++i)
	{
		auto p = prob.data->ptr[i];

		out.data->ptr[i] = -p * log(p) - (1 - p) * log(1 - p);
	}
}

template<typename mode, typename Dtype>
Entropy<mode, Dtype>::Entropy(std::string _name, PropErr _properr) 
				: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void Entropy<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 				std::vector< std::shared_ptr<Variable> >& outputs, 
						 				Phase phase)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& probs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	
	CalcEntropy(probs, output);
}

template<typename mode, typename Dtype>
void Entropy<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
										std::vector< bool >& isConst, 
						 				std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType());	
	if (isConst[0])
		return;
	auto& probs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	ASSERT(probs.cols() == 1, "only support binary now");
	auto grad_input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();

	auto grad_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();
	
	DTensor<mode, Dtype> tmp;
	tmp.CopyFrom(probs);
	tmp.Scale(-1.0);
	tmp.Add(1.0);
	tmp.ElewiseDiv(probs);
	tmp.Log();

	tmp.ElewiseMul(grad_out);
	grad_input.Axpy(1.0, tmp);
}

INSTANTIATE_CLASS(Entropy)

}