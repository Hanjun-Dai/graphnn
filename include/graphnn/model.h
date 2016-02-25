#ifndef MODEL_H
#define MODEL_H

#include "i_param.h"
#include <map>

template<MatMode mode, typename Dtype>
class Model
{
public:

        Model()
        {
            diff_params.clear();
            const_params.clear();
        }   
        
        template<template <MatMode, typename> class ParamType, typename... Args>    
        IDiffParam<mode, Dtype>* add_diff(Args&&... args)
        {
            auto param_name = fmt::sprintf("diff-param-%d", diff_params.size());
            assert(diff_params.count(param_name) == 0);
            
            IDiffParam<mode, Dtype>* param = new ParamType<mode, Dtype>(param_name, std::forward<Args>(args)...);
            diff_params[param_name] = param;
            return param;           
        }
        
        std::map< std::string, IDiffParam<mode, Dtype>* > diff_params;
        std::map< std::string, IConstParam<mode, Dtype>* > const_params;                
};

#endif