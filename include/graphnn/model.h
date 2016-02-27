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
        
        inline void AddParam(IDiffParam<mode, Dtype>* param)
        {
            flatten = false;
            assert(diff_params.count(param->name) == 0);     
            diff_params[param->name] = param;       
        }
               
        std::map< std::string, PP<mode, Dtype>* >& GetDiffParams()
        {
            if (!flatten)
                DiffParams2List();
            return param_list; 
        }
        
        void DiffParams2List()
        {
            param_list.clear();
            for (auto& param_pair : diff_params)
            {
                for (auto& weight_pair : param_pair.second->p)
                {
                    param_list[param_pair.first + "-" + weight_pair.first] = weight_pair.second;
                }
            }
        }
        
        std::map< std::string, IDiffParam<mode, Dtype>* > diff_params;
        std::map< std::string, IConstParam<mode, Dtype>* > const_params;
        bool flatten;                  
        std::map< std::string, PP<mode, Dtype>* > param_list;
};

template<template <MatMode, typename> class ParamType, MatMode mode, typename Dtype, typename... Args>    
IDiffParam<mode, Dtype>* add_diff(Model<mode, Dtype>& model, std::string param_name, Args&&... args)
{
        auto* param = new ParamType<mode, Dtype>(param_name, std::forward<Args>(args)...);
        model.AddParam(param);                                            
        return param;
}

#endif