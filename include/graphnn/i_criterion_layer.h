#ifndef I_CRITERION_LAYER_H
#define I_CRITERION_LAYER_H

#include "i_layer.h"

template<MatMode mode, typename Dtype>
class ICriterionLayer : public ILayer<mode, Dtype>
{
public:
		ICriterionLayer(std::string _name, PropErr _properr = PropErr::T) 
            : ICriterionLayer<mode, Dtype>(_name, 1.0, _properr) {}
            
		ICriterionLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T) 
            : ILayer<mode, Dtype>(_name, _properr), lambda(_lambda), loss(0.0) {}
               
		Dtype GetLoss()
        {
            return loss;    
        }			
        
        virtual bool IsSupervised() override
        {
            return true;
        }
        
        Dtype lambda;
        
protected:        
        Dtype loss;
};

#endif