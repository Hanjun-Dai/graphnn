#ifndef LEARNER_H
#define LEARNER_H

#include "model.h"

template<MatMode mode, typename Dtype>
class Trainer
{
public:
        explicit Trainer(Model<mode, Dtype>* m)
            : model(m)
        {
                
        }                
        
        Model<mode, Dtype>* model; 
};

#endif