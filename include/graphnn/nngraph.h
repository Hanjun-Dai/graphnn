#ifndef NNGRAPH_H
#define NNGRAPH_H

#include "i_layer.h"
#include "i_param.h"
#include "fmt/format.h"
#include <vector>

template<MatMode mode, typename Dtype>
class NNGraph
{
public:
    NNGraph()
    {
        layer_dict.clear();
        static_layer_dict.clear();
        ordered_layers.clear();
        name_idx_map.clear();
        hash.clear();
    }
    
    void Clear()
    {
        layer_dict.clear();
        static_layer_dict.clear();
        ordered_layers.clear();
        name_idx_map.clear();
        hash.clear();
    }
    
    void FeedForward(std::map<std::string, IMatrix<mode, Dtype>* > input, Phase phase);   
       
    std::map<std::string, Dtype> GetLoss();
            
    void BackPropagation();
    
    inline void InsertLayer(ILayer<mode, Dtype>* layer)
    {
        InsertLayer(layer, {});
    }
    
    inline void PrintComputationalGraph()
    {
        for (size_t i = 0; i < ordered_layers.size(); ++i)
        {
            auto* cur_layer = layer_dict[ordered_layers[i].first];
            auto& operands = ordered_layers[i].second;
            
            std::cerr << "( ";
            for (auto* layer : operands)
                std::cerr << layer->name << " ";
            std::cerr << ") -> " << cur_layer->name << std::endl;
        }
    }
    
    inline bool HasLayer(std::string name)
    {
        return static_layer_dict.count(name) || layer_dict.count(name);
    }
    
    inline void InsertStaticLayer(ILayer<mode, Dtype>* layer)
    {
        assert(static_layer_dict.count(layer->name) == 0);
        assert(layer_dict.count(layer->name) == 0);
        static_layer_dict[layer->name] = layer;
        assert(layer->state);
        assert(layer->state->count);
    }
    
    inline void InsertLayer(ILayer<mode, Dtype>* layer, std::vector< ILayer<mode, Dtype>* > operands)
    {
        assert(layer_dict.count(layer->name) == 0);
        layer_dict[layer->name] = layer;
        name_idx_map[layer->name] = ordered_layers.size();
        for (auto* op : operands)
        {
            if (!HasLayer(op->name))
                InsertStaticLayer(op);
        }
        ordered_layers.push_back(std::make_pair(layer->name, operands));        
    }                          
    
    template<MatMode anotherMode>
    void GetState(std::string layer_name, DenseMat<anotherMode, Dtype>& dst)
    {
        assert(layer_dict.count(layer_name));
        auto& output = layer_dict[layer_name]->state->DenseDerived();
        dst.CopyFrom(output);
    }
    
    std::map< std::string, unsigned > name_idx_map;
    std::map< std::string, ILayer<mode, Dtype>* > layer_dict, static_layer_dict;
    std::vector< std::pair<std::string, std::vector< ILayer<mode, Dtype>* > > > ordered_layers;
    std::vector< bool > hash;    
};

template<template <MatMode, typename> class LayerType, MatMode mode, typename Dtype, typename... Args>
inline ILayer<mode, Dtype>* cl(NNGraph<mode, Dtype>& gnn,
                               std::vector< ILayer<mode, Dtype>* > operands, Args&&... args)
{
        return cl<LayerType>(fmt::format("{0}-layer-{1}", LayerType<mode, Dtype>::str_type(), gnn.layer_dict.size()),
                             gnn, 
                             operands, 
                             std::forward<Args>(args)...);
}
    
template<template <MatMode, typename> class LayerType, MatMode mode, typename Dtype, typename... Args>
inline ILayer<mode, Dtype>* cl(const std::string layer_name, 
                               NNGraph<mode, Dtype>& gnn,                                
                               std::vector< ILayer<mode, Dtype>* > operands, 
                               Args&&... args)
{        
        auto* layer = new LayerType<mode, Dtype>(layer_name, std::forward<Args>(args)...);
        gnn.InsertLayer(layer, operands);
        return layer;
}

template<template <MatMode, typename> class LayerType, MatMode mode, typename Dtype, typename... Args>
inline ILayer<mode, Dtype>* cl(NNGraph<mode, Dtype>& gnn, 
                               std::vector< ILayer<mode, Dtype>* > operands,
                               std::vector< IParam<mode, Dtype>* > params, 
                               Args&&... args)
{        
        return cl<LayerType>(fmt::format("{0}-layer-{1}", LayerType<mode, Dtype>::str_type(), gnn.layer_dict.size()),
                             gnn, 
                             operands, 
                             params,                               
                             std::forward<Args>(args)...);
}

// workaround for deducting list
template<template <MatMode, typename> class LayerType, MatMode mode, typename Dtype, typename... Args>
inline ILayer<mode, Dtype>* cl(const std::string layer_name,
                               NNGraph<mode, Dtype>& gnn,
                               std::vector< ILayer<mode, Dtype>* > operands,
                               std::vector< IParam<mode, Dtype>* > params, 
                               Args&&... args)
{
        auto* layer = new LayerType<mode, Dtype>(layer_name, params, std::forward<Args>(args)...);
        gnn.InsertLayer(layer, operands);
        return layer;
}

#endif
