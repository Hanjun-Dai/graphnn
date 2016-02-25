#include "linear_param.h"
#include "dense_matrix.h"
#include <cmath>

template<MatMode mode, typename Dtype>		
void LinearParam<mode, Dtype>::Reset(Dtype mean, Dtype std)
{
	this->p["weight"]->value.SetRandN(mean, std, input_size, output_size);
	this->p["weight"]->grad.Zeros(input_size, output_size);
	if (bo == BiasOption::BIAS)
	{
		this->p["bias"]->value.Zeros(1, output_size);
		this->p["bias"]->grad.Zeros(1, output_size);
	}
}

template class LinearParam<CPU, double>;
template class LinearParam<CPU, float>;
template class LinearParam<GPU, double>;
template class LinearParam<GPU, float>;