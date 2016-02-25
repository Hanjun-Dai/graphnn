#ifndef MODEL_H
#define MODEL_H

#include "i_param.h"
#include <map>
#include <vector>
#include "cppformat/format.h"

template<MatMode mode, typename Dtype>
class Model
{
public:

        Model()
        {
            flatten = false;
            diff_params.clear();
            const_params.clear();
            param_list.clear();
        }
        
        template<template <MatMode, typename> class ParamType, typename... Args>    
        IDiffParam<mode, Dtype>* add_diff(Args&&... args)
        {
            flatten = false;
            auto param_name = fmt::sprintf("diff-param-%d", diff_params.size());
            assert(diff_params.count(param_name) == 0);
            
            IDiffParam<mode, Dtype>* param = new ParamType<mode, Dtype>(param_name, std::forward<Args>(args)...);
            diff_params[param_name] = param;
            return param;           
        }
               
        std::vector< PP<mode, Dtype>* >& GetDiffParams()
        {
            if (!flatten)
                DiffParams2List();
            return param_list; 
        }
        
        void DiffParams2List()
        {
            param_list.clear();
            
        }
        
private:
        std::map< std::string, IDiffParam<mode, Dtype>* > diff_params;
        std::map< std::string, IConstParam<mode, Dtype>* > const_params;
        bool flatten;                  
        std::vector< PP<mode, Dtype>* > param_list;
};

#endif