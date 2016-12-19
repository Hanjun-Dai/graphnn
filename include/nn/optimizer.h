#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "nn/param_set.h"

namespace gnn
{

// template<MatMode mode, typename Dtype>
// class ILearner
// {
// public:
//         explicit ILearner(Model<mode, Dtype>* m, Dtype _init_lr, Dtype _l2_penalty = 0)
//             : model(m), init_lr(_init_lr), l2_penalty(_l2_penalty), cur_lr(_init_lr), clip_threshold(5), clipping_enabled(true), cur_iter(0) {}
                
//         virtual void Update() = 0;                     
        
//         Dtype ClipGradients();

//         Model<mode, Dtype>* model; 
//         Dtype init_lr, l2_penalty, cur_lr;        
//         Dtype clip_threshold;
//         bool clipping_enabled;
//         int cur_iter;
// };


}

#endif