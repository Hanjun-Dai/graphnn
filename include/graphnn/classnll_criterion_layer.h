#ifndef CLASSNLL_CRITERION_LAYER_H
#define CLASSNLL_CRITERION_LAYER_H

#include "i_criterion_layer.h"
#include "dense_matrix.h"
#include "sparse_matrix.h"

template<MatMode mode, typename Dtype>
class ClassNLLCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			ClassNLLCriterionLayer(std::string _name, bool _need_softmax, PropErr _properr = PropErr::T)
                : ClassNLLCriterionLayer<mode, Dtype>(_name, _need_softmax, 1.0, _properr) {}
                
			ClassNLLCriterionLayer(std::string _name, bool _need_softmax, Dtype _lambda, PropErr _properr = PropErr::T)
				 : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr), need_softmax(_need_softmax)
            {
                
            }            
            
			virtual Dtype GetLoss(IMatrix<mode, Dtype>* ground_truth) override
            {
                return 0;
            }
            
protected:
            const bool need_softmax;
            DenseMat<mode, Dtype> buf;
};

#endif