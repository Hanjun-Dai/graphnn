#include "nn/cross_entropy.h"
#include <cmath>

namespace gnn
{

template<typename Dtype>
void CalcCrossEntropy(DTensor<CPU, Dtype>& prob, SpTensor<CPU, Dtype>& label, DTensor<CPU, Dtype>& out)
{
	ASSERT(prob.cols() == label.cols(), "# class doesn't match");
	out.Reshape({prob.rows()});
	for (size_t i = 0; i < prob.rows(); ++i)
	{
		Dtype loss = 0.0;
		for (int k = label.data->row_ptr[i]; k < label.data->row_ptr[i + 1]; ++k)
		    loss -= log(prob.data->ptr[label.cols() * i + label.data->col_idx[k]]) * label.data->val[k];
		out.data->ptr[i] = loss;
	}
}

template<typename mode, typename Dtype>
CrossEntropy<mode, Dtype>::CrossEntropy(std::string _name, bool _need_softmax, PropErr _properr) 
				: Factor(_name, _properr), need_softmax(_need_softmax)
{

}

template<typename mode, typename Dtype>
void CrossEntropy<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 				std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto* lhs = dynamic_cast< TensorVar<mode, Dtype>* >(operands[0].get());
	ASSERT(lhs->GetMatType() == MatType::dense, "[dense prediction] cross_entropy [sparse/dense label]");

	probs.CopyFrom(lhs->template Derived<DENSE>().value);
	if (need_softmax)
		probs.Softmax();
	auto& label = dynamic_cast<SpTensorVar<mode, Dtype>*>(operands[1].get())->value;
	CalcCrossEntropy(probs, label, output);
}

template class CrossEntropy<CPU, float>;
template class CrossEntropy<CPU, double>;


}