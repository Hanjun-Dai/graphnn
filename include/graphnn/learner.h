#ifndef LEARNER_H
#define LEARNER_H

#include "model.h"

template<MatMode mode, typename Dtype>
class ILearner
{
public:
        explicit ILearner(Model<mode, Dtype>* m, Dtype _init_lr, Dtype _l2_penalty = 0)
            : model(m), init_lr(_init_lr), l2_penalty(_l2_penalty), cur_lr(_init_lr), clip_threshold(5), clipping_enabled(true), cur_iter(0) {}
                
        virtual void Update() = 0;                     
        
        Dtype ClipGradients();

        Model<mode, Dtype>* model; 
        Dtype init_lr, l2_penalty, cur_lr;        
        Dtype clip_threshold;
        bool clipping_enabled;
        int cur_iter;
};

template<MatMode mode, typename Dtype>
class SGDLearner : public ILearner<mode, Dtype>
{
public:
        explicit SGDLearner(Model<mode, Dtype>* m, Dtype _init_lr, Dtype _l2_penalty = 0)
                : ILearner<mode, Dtype>(m, _init_lr, _l2_penalty) {}        
        
        virtual void Update() override;                          
};

template<MatMode mode, typename Dtype>
class MomentumSGDLearner : public ILearner<mode, Dtype>
{
public:
        explicit MomentumSGDLearner(Model<mode, Dtype>* m, 
                                    Dtype _init_lr, 
                                    Dtype _momentum = 0.9, 
                                    Dtype _l2_penalty = 0)
                : ILearner<mode, Dtype>(m, _init_lr, _l2_penalty), momentum(_momentum) 
                {
                    acc_grad_dict.clear();
                }
        
        virtual void Update() override;                          
        Dtype momentum;
        std::map<std::string, std::shared_ptr< DenseMat<mode, Dtype> > > acc_grad_dict;
};

template<MatMode mode, typename Dtype>
class ExplicitBatchLearner : public ILearner<mode, Dtype>
{
public:
        explicit ExplicitBatchLearner(Model<mode, Dtype>* m, Dtype _init_lr, Dtype _l2_penalty = 0)
                : ILearner<mode, Dtype>(m, _init_lr, _l2_penalty) 
                {
                    acc_grad_dict.clear();
                }
        
        virtual void Update() override;      
        void AccumulateGrad();                    
        
        std::map<std::string, std::shared_ptr< DenseMat<mode, Dtype> > > acc_grad_dict;
};

template<MatMode mode, typename Dtype>
class AdamLearner : public ILearner<mode, Dtype>
{
public:
        explicit AdamLearner(Model<mode, Dtype>* m, 
                            Dtype _init_lr,
                            Dtype _l2_penalty = 0, 
                            Dtype _beta_1 = 0.9, 
                            Dtype _beta_2 = 0.999, 
                            Dtype _eps = 1e-8)
                : ILearner<mode, Dtype>(m, _init_lr, _l2_penalty), beta_1(_beta_1), beta_2(_beta_2), eps(_eps)
                {
                    first_moments.clear();
                    second_moments.clear();
                }

        virtual void Update() override;

        std::map<std::string, std::shared_ptr< DenseMat<mode, Dtype> > > first_moments, second_moments;
        Dtype beta_1, beta_2, eps;
        DenseMat<mode, Dtype> m_hat, v_hat;
};

#endif