#ifndef I_CRITERION_LAYER_H
#define I_CRITERION_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class ICriterionLayer : public ILayer<mode, Dtype>
{
public:
		ICriterionLayer(std::string _name, PropErr _properr = PropErr::T) 
            : ILayer<mode, Dtype>(_name, _properr), lambda(1.0) {}
            
		ICriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T) 
            : ILayer<mode, Dtype>(_name, _properr), lambda(_lambda) {}
        
		virtual Dtype GetLoss(IMatrix<mode, Dtype>* ground_truth) = 0;			
        
        Dtype lambda;
};

#endif