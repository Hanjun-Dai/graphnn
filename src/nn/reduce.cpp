#include "nn/reduce.h"

namespace gnn
{


template<typename mode, typename Dtype>
Reduce<mode, Dtype>::Reduce(std::string _name, ReduceType _r_type, int _axis, bool _keep_dim, PropErr _properr) 
				: Factor(_name, _properr), r_type(_r_type), axis(_axis), keep_dim(_keep_dim)
{

}

template<typename mode, typename Dtype>
void Reduce<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
						 			Phase phase)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto& input = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;

	ASSERT(axis == -1 && keep_dim == false, "currently only support axis=-1 and keep_dim=false in " << StrType());

	switch (r_type)
	{
		case ReduceType::MEAN: 
			output.Mean(input, axis);
			break;
		case ReduceType::SUM:
			output.Sum(input, axis);
			break;
		default:
			ASSERT(false, "unexpected reduction type");
			break;
	}

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
void Reduce<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
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

	switch (r_type)
	{
		case ReduceType::MEAN: 
			input.Add(grad_out / input.shape.Count());
			break;
		case ReduceType::SUM:
			input.Add(grad_out);
			break;
		default:
			ASSERT(false, "unexpected reduction type");
			break;
	}
}

INSTANTIATE_CLASS(Reduce)

}