#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "nn/param_set.h"

namespace gnn
{

/**
 * @brief      abstract class for optimizer.
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class IOptimizer
{
public:
	/**
	 * @brief      constructor
	 *
	 * @param      _param_set   The parameter set
	 * @param[in]  _init_lr     The initial learning rate
	 * @param[in]  _l2_penalty  The l2 penalty coeff
	 */
	IOptimizer(ParamSet<mode, Dtype>* _param_set, Dtype _init_lr, Dtype _l2_penalty = 0); 

	/**
	 * @brief      update the parameters
	 */
	virtual void Update() = 0;

	/**
	 * @brief      clipping by rescaling
	 *
	 * @return     scale of gradient; if no clipping, 1.0 is returned
	 */
	Dtype ClipGradients();

	/**
	 * the bind paramters to this optimizer
	 */
	ParamSet<mode, Dtype>* param_set; 

	/**
	 * initial learning rate
	 */
	Dtype init_lr;
	/**
	 * coeff of l2-penalty, default is 0
	 */
	Dtype l2_penalty;
	/**
	 * current learning rate
	 */
	Dtype cur_lr;
	/**
	 * threshold of gradient clipping, default is 5.0
	 */
	Dtype clip_threshold;

	/**
	 * whether enables the gradient clipping; default is true
	 */
	bool clipping_enabled;

	/**
	 * the index of current iteration
	 */
	int cur_iter;
};

/**
 * @brief      Class for simple sgd optimizer.
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class SGDOptimizer: public IOptimizer<mode, Dtype>
{
public:

	/**
	 * @brief      constructor
	 *
	 * @param      m            The parameter set
	 * @param[in]  _init_lr     The initialize lr
	 * @param[in]  _l2_penalty  The l2 penalty coeff
	 */
	SGDOptimizer(ParamSet<mode, Dtype>* m, Dtype _init_lr, Dtype _l2_penalty = 0);

	virtual void Update() override;
};

/**
 * @brief      Class for momentum sgd optimizer.
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class MomentumSGDOptimizer: public IOptimizer<mode, Dtype>
{
public:

	/**
	 * @brief      constructor
	 *
	 * @param      m            The parameter set
	 * @param[in]  _init_lr     The initialize lr
	 * @param[in]  _momentum    The momentum
	 * @param[in]  _l2_penalty  The l2 penalty coeff
	 */
	MomentumSGDOptimizer(ParamSet<mode, Dtype>* m, Dtype _init_lr, Dtype _momentum = 0.9, Dtype _l2_penalty = 0);

	virtual void Update() override;
	/**
	 * momentum
	 */
	Dtype momentum;
	/**
	 * accumulated gradient
	 */
	std::map<std::string, std::shared_ptr< DTensor<mode, Dtype> > > acc_grad_dict;
};

/**
 * @brief      Class for adam optimizer.
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
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
	RowSpTensor<mode, Dtype> v_hat;
};

}

#endif