#include "nn/kxplusb.h"

namespace gnn
{

template<typename mode, typename Dtype>
Kxplusb<mode, Dtype>::Kxplusb(std::string _name, Dtype _k, Dtype _b, PropErr _properr) 
		: Factor(_name, _properr), k(_k), b(_b)
{

}

template<typename mode, typename Dtype>
void Kxplusb<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& y = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& x = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	y.CopyFrom(x);
	y.Scale(k);
	y.Add(b);
}

template<typename mode, typename Dtype>
void Kxplusb<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

	if (isConst[0])
		return;
	auto grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();
	grad.Axpy(k, cur_grad);
}

INSTANTIATE_CLASS(Kxplusb)

}