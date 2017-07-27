#include "nn/inner_product.h"

namespace gnn
{
template<typename mode, typename Dtype>
InnerProduct<mode, Dtype>::InnerProduct(std::string _name, PropErr _properr) 
		: Factor(_name, _properr)
{

}

template<typename mode, typename Dtype>
void InnerProduct<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	auto& lhs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
    auto& rhs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;

    buf.CopyFrom(lhs);
    buf.ElewiseMul(rhs);

    ones.Reshape({lhs.shape.Count(1), 1});
    ones.Fill(1.0);

    output.MM(buf, ones, Trans::N, Trans::N, 1.0, 0.0);
}

template<typename mode, typename Dtype>
void InnerProduct<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto cur_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

    for (int i = 0; i < 2; ++i)
    {
        if (isConst[i])
            continue;
        auto grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[i].get())->grad.Full();
        auto& another_operand = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1 - i].get())->value;

        buf.CopyFrom(another_operand);
        buf.ElewiseMul(cur_grad);
        grad.Axpy(1.0, buf);
    }
}

INSTANTIATE_CLASS(InnerProduct)

}