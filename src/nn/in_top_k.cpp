#include "nn/in_top_k.h"

namespace gnn
{

template<typename Dtype>
void IsInTopK(DTensor<CPU, Dtype>& pred, DTensor<CPU, int>& label, DTensor<CPU, int>& out, int k)
{
	ASSERT(pred.rank() == 2, "predicted prob(or logits) should be a matrix");
	ASSERT(pred.rows() == label.shape.Count(), "# instances doesn't match");
	out.Reshape(label.shape.dims);
	for (size_t i = 0; i < pred.rows(); ++i)
	{
		Dtype* ptr = pred.data->ptr + i * pred.cols();

		int t = 0, target = label.data->ptr[i];
		for (size_t j = 0; j < pred.cols(); ++j)
			if ((int)j != target && ptr[j] > ptr[target])
				t += 1;
		
		out.data->ptr[i] = t < k;
	}
}

template<typename mode, typename Dtype>
InTopK<mode, Dtype>::InTopK(std::string _name, int _topK) : Factor(_name, PropErr::N), topK(_topK)
{
	
}

template<typename mode, typename Dtype>
void InTopK<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& out_bool = dynamic_cast<DTensorVar<mode, int>*>(outputs[0].get())->value;

	auto& pred = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& label = dynamic_cast<DTensorVar<mode, int>*>(operands[1].get())->value;

	IsInTopK(pred, label, out_bool, topK);
}

INSTANTIATE_CLASS(InTopK)

}
