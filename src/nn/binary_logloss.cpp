#include "nn/binary_logloss.h"
#include <cmath>

namespace gnn
{

template<typename Dtype>
void CalcBinaryLogLoss(DTensor<CPU, Dtype>& prob, DTensor<CPU, Dtype>& label, DTensor<CPU, Dtype>& out)
{
	ASSERT(prob.cols() == label.cols(), "# columns should match");
	out.Reshape({prob.rows(), prob.cols()});
	for (size_t i = 0; i < prob.rows(); ++i)
	{
	        for (size_t j = 0; j < prob.cols(); ++j)
                {
		        auto& y = label.data->ptr[i * prob.cols() + j];
		        auto& p = prob.data->ptr[i * prob.cols() + j];
		        out.data->ptr[i * prob.cols() + j] = -y * log(p + 1e-9) - (1 - y) * log(1 - p + 1e-9);
                }
	}
}

template<typename mode, typename Dtype>
BinaryLogLoss<mode, Dtype>::BinaryLogLoss(std::string _name, bool _need_sigmoid, PropErr _properr) 
				: Factor(_name, _properr), need_sigmoid(_need_sigmoid)
{

}

template<typename mode, typename Dtype>
void BinaryLogLoss<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 				std::vector< std::shared_ptr<Variable> >& outputs, 
						 				Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& raw_pred = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& label = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;
	
	auto& probs = need_sigmoid ? tmp_probs : raw_pred;	
	if (need_sigmoid)
	{
		probs.CopyFrom(raw_pred);
		probs.Sigmoid();
	}
	CalcBinaryLogLoss(probs, label, output);
}

template<typename mode, typename Dtype>
void BinaryLogLoss<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
										std::vector< bool >& isConst, 
						 				std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType());
	if (isConst[0])
		return;
	auto& raw_pred = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& label = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;
	auto& probs = need_sigmoid ? tmp_probs : raw_pred;

	auto grad_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();
	auto grad_lhs = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();

	if (need_sigmoid)
	{
		probs.Axpy(-1.0, label);
		probs.ElewiseMul(grad_out);
		grad_lhs.Axpy(1.0, probs);
	} else 
	{
		DTensor<mode, Dtype> tmp;
		tmp.CopyFrom(probs);
		tmp.Axpy(-1.0, label);
		tmp.ElewiseDiv(probs);
		probs.Scale(-1.0);
		probs.Add(1.0);
		tmp.ElewiseDiv(probs);
		tmp.ElewiseMul(grad_out);

		grad_lhs.Axpy(1.0, tmp);
	}
}

INSTANTIATE_CLASS(BinaryLogLoss)

}
