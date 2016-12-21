#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "nn/param_set.h"

namespace gnn
{

template<typename mode, typename Dtype>
class IOptimizer
{
public:
	IOptimizer(ParamSet<mode, Dtype>* _param_set, Dtype _init_lr, Dtype _l2_penalty = 0); 

	virtual void Update() = 0;
	Dtype ClipGradients();

	ParamSet<mode, Dtype>* param_set; 
	Dtype init_lr, l2_penalty, cur_lr;
	Dtype clip_threshold;
	bool clipping_enabled;
	int cur_iter;
};

template<typename mode, typename Dtype>
class AdamOptimizer : public IOptimizer<mode, Dtype>
{
public:
	AdamOptimizer(ParamSet<mode, Dtype>* _param_set, 
				Dtype _init_lr,
				Dtype _l2_penalty = 0, 
				Dtype _beta_1 = 0.9, 
				Dtype _beta_2 = 0.999, 
				Dtype _eps = 1e-8);                     

	virtual void Update() override;

	std::map<std::string, std::shared_ptr< DTensor<mode, Dtype> > > first_moments, second_moments;
	Dtype beta_1, beta_2, eps;
	DTensor<mode, Dtype> m_hat, v_hat;
};

}

#endif