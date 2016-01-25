#ifndef GRAPHNN_H
#define GRAPHNN_H

#include "dense_matrix.h"
#include "ilayer.h"
#include "icriterion_layer.h"
#include "iparam.h"
#include "graph_data.h"
#include <map>
#include <string>
#include <vector>

template<MatMode mode, typename Dtype>
class GraphNN
{
public:
		typedef typename std::map<std::string, GraphData<mode, Dtype>* >::iterator dataiter;
		typedef typename std::map<std::string, IParam<mode, Dtype>* >::iterator paramiter;
		typedef typename std::map<std::string, ILayer<mode, Dtype>* >::iterator layeriter;
		
		GraphNN();
		
		void ForwardData(std::map<std::string, GraphData<mode, Dtype>* > graph_input, Phase phase);
		std::map<std::string, Dtype> ForwardLabel(std::map<std::string, GraphData<mode, Dtype>* > graph_truth);
		
		void BackPropagation();
		
		void UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum);
		
		void AddParam(IParam<mode, Dtype>* param);
		void AddLayer(ILayer<mode, Dtype>* layer);
		void AddEdge(std::string layername_from, std::string layername_to);
		void AddEdge(ILayer<mode, Dtype>* layerfrom, ILayer<mode, Dtype>* layerto);		
		
		std::map< std::string, IParam<mode, Dtype>* > param_dict;
		std::map< std::string, ILayer<mode, Dtype>* > layer_dict;
		std::vector< std::pair<std::string, std::string> > sorted_edges;
			
		void Save(std::string filename);
		void Load(std::string filename);			
		
        template<MatMode anotherMode>
        void GetDenseNodeState(std::string layer_name, DenseMat<anotherMode, Dtype>& dst)
        {
            auto& output = layer_dict[layer_name]->graph_output->node_states->DenseDerived();
            dst.CopyFrom(output);
        }
        
		GraphStruct<std::string> layer_graph;
		void InitializeGraph();
private: 

		void Serialize(FILE* fid);
		void Deserialize(FILE* fid);
		bool initialized;
		bool* visited, *has_grad;
		std::map< std::string, int > name_idx_map;
};

#endif