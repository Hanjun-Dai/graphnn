#include "tensor/unary_functor.h"
#include <chrono>

namespace gnn
{

template<typename Dtype>
UnarySet<CPU, Dtype>::UnarySet(Dtype _scalar) : scalar(_scalar) {}

template<typename Dtype>
void UnarySet<CPU, Dtype>::operator()(Dtype& dst)
{
	dst = scalar;
}

template class UnarySet<CPU, float>;
template class UnarySet<CPU, double>;
template class UnarySet<CPU, int>;

template<typename Dtype>
UnaryRandNorm<CPU, Dtype>::UnaryRandNorm(Dtype _mean, Dtype _std) : mean(_mean), std(_std) 
{
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	generator = new std::default_random_engine(seed);

	distribution = new std::normal_distribution<Dtype>(mean, std);
}

template<typename Dtype>
void UnaryRandNorm<CPU, Dtype>::operator()(Dtype& dst)
{
	auto& dist = *distribution;
	auto& engine = *generator;
	dst = dist(engine);
}

template class UnaryRandNorm<CPU, float>;
template class UnaryRandNorm<CPU, double>;

}