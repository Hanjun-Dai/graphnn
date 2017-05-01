#include "nn/hit_at_k.h"
#include <set>

namespace gnn
{

template<typename Dtype>
void HitsTopK(DTensor<CPU, Dtype>& pred, SpTensor<CPU, Dtype>& label, DTensor<CPU, int>& out, int k)
{
	ASSERT(pred.rank() == 2, "predicted prob(or logits) should be a matrix");
	ASSERT(pred.rows() == label.rows(), "# instances doesn't match");
	if (k > (int)pred.cols())
		k = pred.cols();

	out.Reshape({pred.rows(), 1});
	std::set<int> topk_idxes;
	for (size_t i = 0; i < pred.rows(); ++i)
	{
		Dtype* ptr = pred.data->ptr + i * pred.cols();
		topk_idxes.clear();

		for (int cur_k = 0; cur_k < k; ++cur_k)
		{
			int cur_best = -1;
			for (size_t j = 0; j < pred.cols(); ++j)
				if (cur_best < 0 || ptr[j] > ptr[cur_best])
				{
					if (!topk_idxes.count(j))
						cur_best = j;
				}
			ASSERT(cur_best >= 0, "unexpected behavior in hit@k");
			topk_idxes.insert(cur_best);
		}
		bool hit = false;
		for (int j = label.data->row_ptr[i]; j < label.data->row_ptr[i + 1]; ++j)
			if (topk_idxes.count(label.data->col_idx[j]))
			{
				hit = true;
				break;
			}
		
		out.data->ptr[i] = hit;
	}
}

template<typename mode, typename Dtype>
HitAtK<mode, Dtype>::HitAtK(std::string _name, int _topK) : Factor(_name, PropErr::N), topK(_topK)
{
	
}

template<typename mode, typename Dtype>
void HitAtK<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& out_bool = dynamic_cast<DTensorVar<mode, int>*>(outputs[0].get())->value;

	auto& pred = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& label = dynamic_cast<SpTensorVar<mode, Dtype>*>(operands[1].get())->value;

	HitsTopK(pred, label, out_bool, topK);
}

INSTANTIATE_CLASS(HitAtK)

}
