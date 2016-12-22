#ifndef MODEL_H
#define MODEL_H

#include "i_param.h"
#include <map>
#include <vector>
#include "fmt/format.h"

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
            all_params[param->name] = param;       
        }
        
        inline void AddParam(IConstParam<mode, Dtype>* param)
        {
            assert(const_params.count(param->name) == 0);
            const_params[param->name] = param;
            all_params[param->name] = param;
        }
        
        inline void SetupConstParams(std::map<std::string, void*> arg_dict)
        {
            for (auto& p : arg_dict)
            {
                assert(const_params.count(p.first));
                const_params[p.first]->InitConst(p.second);
            }
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
        void Load(std::string filename)
        {
            FILE* fid = fopen(filename.c_str(), "rb");
            
            for (auto it = diff_params.begin(); it != diff_params.end(); ++it)
                it->second->Deserialize(fid); 
                
            fclose(fid);
        }
       
        void Save(std::string filename)
        {
            FILE* fid = fopen(filename.c_str(), "wb");
            
            for (auto it = diff_params.begin(); it != diff_params.end(); ++it)
                it->second->Serialize(fid); 
                
            fclose(fid);            
        }
        
        std::map< std::string, IDiffParam<mode, Dtype>* > diff_params;
        std::map< std::string, IConstParam<mode, Dtype>* > const_params;
        std::map< std::string, IParam<mode, Dtype>*> all_params;
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

template<template <MatMode, typename> class ParamType, MatMode mode, typename Dtype, typename... Args>    
IConstParam<mode, Dtype>* add_const(Model<mode, Dtype>& model, std::string param_name, Args&&... args)
{
        auto* param = new ParamType<mode, Dtype>(param_name, std::forward<Args>(args)...);
        model.AddParam(param);                                            
        return param;
}

#endif
