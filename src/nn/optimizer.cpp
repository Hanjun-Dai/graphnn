#include "nn/optimizer.h"
#include <cassert>
#include <cmath>

namespace gnn
{

template<typename mode, typename Dtype>
IOptimizer<mode, Dtype>::IOptimizer(ParamSet<mode, Dtype>* _param_set, Dtype _init_lr, Dtype _l2_penalty)
	: param_set(_param_set), init_lr(_init_lr), l2_penalty(_l2_penalty), cur_lr(_init_lr), clip_threshold(5), clipping_enabled(true), cur_iter(0)
{
}

template<typename mode, typename Dtype>
Dtype IOptimizer<mode, Dtype>::ClipGradients()
{
	if (this->clipping_enabled)
	{		        
		Dtype norm = 0.0;
		for (auto& map_pair : param_set->params)
		{
			auto p = map_pair.second;		    
		    Dtype norm2 = p->grad.Norm2();
		    norm += norm2 * norm2;
		}
		norm = sqrt(norm);
		if (norm > this->clip_threshold)
		    return this->clip_threshold / norm;
	}
	return 1.0;	
}

template<typename mode, typename Dtype>
AdamOptimizer<mode, Dtype>::AdamOptimizer(ParamSet<mode, Dtype>* _param_set, Dtype _init_lr, Dtype _l2_penalty, 
				Dtype _beta_1, Dtype _beta_2, Dtype _eps) : IOptimizer<mode, Dtype>(_param_set, _init_lr, _l2_penalty), 
															beta_1(_beta_1), beta_2(_beta_2), eps(_eps)
{
	first_moments.clear();
	second_moments.clear();
}

template<typename mode, typename Dtype>
void AdamOptimizer<mode, Dtype>::Update()
{            
    Dtype gscale = this->ClipGradients();
    this->cur_iter++;

    for (auto& param_pair : this->param_set->params)
    {
		auto name = param_pair.first;
		auto param = param_pair.second;

		if (first_moments.count(name) == 0 && second_moments.count(name) == 0)
		{
		    first_moments[name] = std::make_shared< DTensor<mode, Dtype> >(param->grad.shape);
		    first_moments[name]->Zeros();
		    second_moments[name] = std::make_shared< DTensor<mode, Dtype> >(param->grad.shape);
		    second_moments[name]->Zeros();
		}
		assert(first_moments.count(name) && second_moments.count(name));
		auto& m_t = *(first_moments[name]); 
		auto& v_t = *(second_moments[name]);
		// clipping and weight decay
		param->grad.Axpby(this->l2_penalty, param->value, gscale);
		// m_t = beta_1 * m_{t-1} + (1 - beta_1) * gt
		m_t.Axpby(1 - beta_1, param->grad, beta_1);
		// v_t = beta_2 * v_{t-1} + (1 - beta_2) * gt^2
		param->grad.Square();
		v_t.Axpby(1 - beta_2, param->grad, beta_2);

		// 1 / (1 - beta^t)
		Dtype s1 = 1.0 / (1 - pow(beta_1, this->cur_iter));
		Dtype s2 = 1.0 / (1 - pow(beta_2, this->cur_iter)); 

		// v_hat = 1 ./ (sqrt(v_t / (1 - beta_2^t)) + eps)
		v_hat.CopyFrom(v_t);
		v_hat.Scale(s2);
		v_hat.Sqrt();
		v_hat.Add(eps);
		v_hat.Inv();

		v_hat.ElewiseMul(m_t);
		param->value.Axpby(-this->cur_lr * s1, v_hat, 1.0);

		param->grad.Zeros();
    }
}

template class AdamOptimizer<CPU, float>;
template class AdamOptimizer<CPU, double>;

}