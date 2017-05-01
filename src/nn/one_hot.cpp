#include "nn/one_hot.h"

namespace gnn
{

template<typename mode, typename Dtype>
OneHot<mode, Dtype>::OneHot(std::string _name, size_t _dim) 
					: Factor(_name, PropErr::N), dim(_dim)
{
}

template<typename mode, typename Dtype>
void OneHot<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
								 			std::vector< std::shared_ptr<Variable> >& outputs, 
								 			Phase phase)
{
	ASSERT(operands.size() == 1 || operands.size() == 2, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& input = dynamic_cast<DTensorVar<mode, int>*>(operands[0].get())->value;
	auto& output = dynamic_cast<SpTensorVar<mode, Dtype>*>(outputs[0].get())->value;

	ASSERT(input.cols() == 1, "only support column vector to sparse one hot matrix");

	output.Reshape({input.rows(), dim});
	output.ResizeSp(input.rows(), input.rows() + 1);

	std::vector<int> idxes(input.rows() + 1);
	for (size_t i = 0; i < idxes.size(); ++i)
		idxes[i] = i;

	DTensor<mode, Dtype> values(input.shape, output.data->val);
	if (operands.size() == 1)		
		values.Fill(1.0);
	else {
		auto& weights = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;
		ASSERT(weights.rows() == input.rows() && weights.cols() == input.cols(), "index and weight size not match");
		values.CopyFrom(weights);
	}

	if (mode::type == MatMode::cpu)
	{
		memcpy(output.data->row_ptr, idxes.data(), sizeof(int) * idxes.size());
		memcpy(output.data->col_idx, input.data->ptr, sizeof(int) * input.rows());
	} 
#ifdef USE_GPU
	else {
		cudaMemcpy(output.data->row_ptr, idxes.data(), sizeof(int) * idxes.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(output.data->col_idx, input.data->ptr, sizeof(int) * input.rows(), cudaMemcpyDeviceToDevice);
	}
#endif
}

INSTANTIATE_CLASS(OneHot)

}