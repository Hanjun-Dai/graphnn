#include "nn/multinomial_sample.h"

namespace gnn
{

template<typename mode, typename Dtype>
MultinomialSample<mode, Dtype>::MultinomialSample(std::string _name, bool _need_softmax)
		: Factor(_name, PropErr::N), need_softmax(_need_softmax)
{
	distribution = new std::uniform_real_distribution<Dtype>(0.0, 1.0);
}

template<typename mode, typename Dtype>
void MultinomialSample<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
								 			std::vector< std::shared_ptr<Variable> >& outputs, 
								 			Phase phase)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& output = dynamic_cast<DTensorVar<CPU, int>*>(outputs[0].get())->value;

	DTensor<CPU, Dtype> probs;
	probs.CopyFrom(input);
	if (need_softmax)
		probs.Softmax();

	output.Reshape({input.rows(), (size_t)1});
	
	for (size_t i = 0; i < probs.rows(); ++i)
	{
		Dtype sum = 0;
		Dtype threshold = (*distribution)(generator);
		int idx = -1;

		for (size_t j = 0; j < probs.cols(); ++j)
		{
			sum += probs.data->ptr[i * probs.cols() + j];
			if (sum >= threshold)
			{
				idx = j;
				break;
			}
		}
		if (idx < 0)
			idx = (int)probs.cols() - 1;
		ASSERT(idx >= 0 && idx < (int)probs.cols(), "unexpected summation of probs");
		output.data->ptr[i] = idx;
	}
}

INSTANTIATE_CLASS(MultinomialSample)

}