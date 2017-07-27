#include "nn/reduce_mean.h"

namespace gnn
{


template<typename mode, typename Dtype>
ReduceMean<mode, Dtype>::ReduceMean(std::string _name, int _axis, bool _keep_dim, PropErr _properr) 
				: Factor(_name, _properr), axis(_axis), keep_dim(_keep_dim)
{

}

template<typename mode, typename Dtype>
void ReduceMean<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	ASSERT(axis == -1 && keep_dim == false, "currently only support axis=-1 and keep_dim=false in " << StrType());

	output.Mean(input, axis);

	auto out_shape = input.shape;
	if (axis == -1)
	{
		if (keep_dim)
		{
			for (size_t i = 0; i < out_shape.dims.size(); ++i)
				out_shape.dims[i] = 1;
		} else
			out_shape.Reshape({1});
	}

	assert(out_shape.Count() == output.shape.Count());
	output.shape = out_shape;
}

template<typename mode, typename Dtype>
void ReduceMean<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 
	if (isConst[0])
		return;
	
	ASSERT(axis == -1 && keep_dim == false, "currently only support axis=-1 and keep_dim=false in " << StrType());

	auto output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();
	auto input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();
	Dtype grad_out = output.AsScalar();
	grad_out /= input.shape.Count();

	input.Add(grad_out);
}

INSTANTIATE_CLASS(ReduceMean)

}