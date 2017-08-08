#include "nn/optimizer.h"
#include <cassert>
#include <cmath>
#include <chrono>
double t_ff = 0.0, t_bp = 0.0;
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

INSTANTIATE_CLASS(IOptimizer)

//==============================================================================

template<typename mode, typename Dtype>
SGDOptimizer<mode, Dtype>::SGDOptimizer(ParamSet<mode, Dtype>* _param_set, 
			Dtype _init_lr, Dtype _l2_penalty) : 
			IOptimizer<mode, Dtype>(_param_set, _init_lr, _l2_penalty)
{
}			

template<typename mode, typename Dtype>
void SGDOptimizer<mode, Dtype>::Update()
{            
    for (auto& param_pair : this->param_set->params)
    {        
        auto& param = param_pair.second;
        param->value.RowSparseAxpby(-this->cur_lr, param->grad, 1 - this->cur_lr * this->l2_penalty);
        param->grad.RowSpZeros();
    }
}

INSTANTIATE_CLASS(SGDOptimizer)

//==============================================================================

template<typename mode, typename Dtype>
MomentumSGDOptimizer<mode, Dtype>::MomentumSGDOptimizer(ParamSet<mode, Dtype>* _param_set, 
			Dtype _init_lr, Dtype _momentum, Dtype _l2_penalty) : 
			IOptimizer<mode, Dtype>(_param_set, _init_lr, _l2_penalty), momentum(_momentum) 
{
	acc_grad_dict.clear();
}			

template<typename mode, typename Dtype>
void MomentumSGDOptimizer<mode, Dtype>::Update()
{            
    for (auto& param_pair : this->param_set->params)
    {        
        auto& name = param_pair.first;
        auto& param = param_pair.second;   
        if (momentum > 0)
        {
            if (acc_grad_dict.count(name) == 0)
            {
                acc_grad_dict[name] = std::make_shared< DTensor<mode, Dtype> >(param->grad.shape);
                acc_grad_dict[name]->Zeros();
            }
            param->grad.RowSparseAxpy(this->l2_penalty, param->value);
            acc_grad_dict[name]->RowSparseAxpby(this->cur_lr, param->grad, momentum);
            if (param->grad.is_full)
            	param->value.Axpy(-1.0, *acc_grad_dict[name]);
            else if (param->grad.row_idxes.shape.Count())
            	param->value.RowSelectiveAxpy(param->grad.row_idxes, -1.0, *acc_grad_dict[name]);
        } else // do normal sgd
            param->value.RowSparseAxpby(-this->cur_lr, param->grad, 1 - this->cur_lr * this->l2_penalty);
        param->grad.RowSpZeros();
    }    
}

INSTANTIATE_CLASS(MomentumSGDOptimizer)

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
		param->grad.RowSparseAxpby(this->l2_penalty, param->value, gscale);
		// m_t = beta_1 * m_{t-1} + (1 - beta_1) * gt
		m_t.RowSparseAxpby(1 - beta_1, param->grad, beta_1);
		// v_t = beta_2 * v_{t-1} + (1 - beta_2) * gt^2
		param->grad.Square();
		v_t.RowSparseAxpby(1 - beta_2, param->grad, beta_2);

		// 1 / (1 - beta^t)
		Dtype s1 = 1.0 / (1 - pow(beta_1, this->cur_iter));
		Dtype s2 = 1.0 / (1 - pow(beta_2, this->cur_iter)); 

		// v_hat = 1 ./ (sqrt(v_t / (1 - beta_2^t)) + eps)

		ASSERT(!v_hat.is_full && v_hat.row_idxes.shape.Count() == 0, "v_hat should be empty at the beginning");
		if (v_hat.data == nullptr || v_t.shape.Count() > v_hat.data->mem_size)
		{
			v_hat.Reshape(v_t.shape.dims);
			v_hat.FullZeros();
		}

		v_hat.ReshapeLike(param->grad);		

		v_hat.RowSparseCopy(v_t);
		v_hat.Scale(s2);

		v_hat.Sqrt();

		v_hat.RowSparseAdd(eps);
		v_hat.RowSparseInv();
		v_hat.ElewiseMul(m_t);

		param->value.RowSparseAxpby(-this->cur_lr * s1, v_hat, 1.0);
		v_hat.RowSpZeros();	
		param->grad.RowSpZeros();
    }
}

INSTANTIATE_CLASS(AdamOptimizer)

}