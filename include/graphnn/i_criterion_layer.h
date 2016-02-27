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
            : ILayer<mode, Dtype>(_name, _properr), lambda(_lambda) {}
        
        virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
        {
            assert(operands.size() == 1); 
            this->state = operands[0]->state;
        } 
        
		virtual Dtype GetLoss(IMatrix<mode, Dtype>* ground_truth) = 0;			
        
        Dtype lambda;
};

#endif