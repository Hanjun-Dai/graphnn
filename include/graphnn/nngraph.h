#ifndef NNGRAPH_H
#define NNGRAPH_H

#include "i_layer.h"
#include "cppformat/format.h"
#include <vector>

template<MatMode mode, typename Dtype>
class NNGraph
{
public:
    NNGraph()
    {
        layer_dict.clear();
        ordered_layers.clear();
        name_idx_map.clear();
        has_grad.clear();
    }
    
    void ForwardData(std::map<std::string, IMatrix<mode, Dtype>* > input, Phase phase);   
       
    std::map<std::string, Dtype> ForwardLabel(std::map<std::string, IMatrix<mode, Dtype>* > ground_truth);
            
    void BackPropagation();
    
    template<template <MatMode, typename> class LayerType, typename... Args>
    inline ILayer<mode, Dtype>* cl(std::vector< ILayer<mode, Dtype>* > operands, Args&&... args)
    {
        return cl<LayerType>(fmt::sprintf("%s-layer-%d", LayerType<mode, Dtype>::str_type(), layer_dict.size()), 
                             operands, 
                             std::forward<Args>(args)...);
    }
    
    template<template <MatMode, typename> class LayerType, typename... Args>
    inline ILayer<mode, Dtype>* cl(std::string layer_name, 
                            std::vector< ILayer<mode, Dtype>* > operands, 
                            Args&&... args)
    {
        assert(layer_dict.count(layer_name) == 0);        
        ILayer<mode, Dtype>* layer = new LayerType<mode, Dtype>(layer_name, std::forward<Args>(args)...);
        layer_dict[layer_name] = layer;
        name_idx_map[layer_name] = ordered_layers.size();
        ordered_layers.push_back(std::make_pair(layer_name, operands));
        return layer;
    }
    
    std::map< std::string, unsigned > name_idx_map;
    std::map< std::string, ILayer<mode, Dtype>* > layer_dict;
    std::vector< std::pair<std::string, std::vector< ILayer<mode, Dtype>* > > > ordered_layers;
    std::vector< bool > has_grad;    
};

#endif