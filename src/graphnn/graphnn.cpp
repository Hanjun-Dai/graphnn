#include "graphnn.h"
#include <cstring>
#include <iostream>

template<MatMode mode, typename Dtype>
GraphNN<mode, Dtype>::GraphNN()
{
	layer_dict.clear();
	param_dict.clear();
	sorted_edges.clear();
	layer_graph.Resize(1);
	initialized = false;
	visited = nullptr;
    has_grad = nullptr;
	name_idx_map.clear();
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::AddLayer(ILayer<mode, Dtype>* layer)
{
	if (layer_dict.count(layer->name) == 0)
	{
		initialized = false;
		layer_dict[layer->name] = layer;
		layer_graph.AddNode(0, layer->name);
	}
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::AddEdge(std::string layername_from, std::string layername_to)
{
	assert(layer_dict.count(layername_from));
	assert(layer_dict.count(layername_to));
	initialized = false;
	layer_graph.AddEdge(layer_graph.num_edges, layername_from, layername_to);
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::AddEdge(ILayer<mode, Dtype>* layerfrom, ILayer<mode, Dtype>* layerto)
{
	AddLayer(layerfrom);
	AddLayer(layerto);
	AddEdge(layerfrom->name, layerto->name);	
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::AddParam(IParam<mode, Dtype>* param)
{
	param_dict[param->name] = param;
}

std::string Att2Str(GraphAtt at)
{
    switch (at)
    {
        case GraphAtt::NODE:
            return "NODE";
        case GraphAtt::EDGE:
            return "EDGE";
        case GraphAtt::NODE_EDGE:
            return "NODE_EDGE";
        default:
            return "unknown";
    }
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::InitializeGraph()
{
    std::cerr << "================== list of params ==================" << std::endl;
    for (paramiter it = param_dict.begin(); it != param_dict.end(); ++it)
        std::cerr << it->second -> name << std::endl; 
    std::cerr << "================== end of param list ==================" << std::endl;
	std::cerr << "initializing" << std::endl;
	layer_graph.TopSort(sorted_edges);
	for (size_t i = 0; i < sorted_edges.size(); ++i)
    {
        std::cerr << sorted_edges[i].first;
        std::cerr << " (" << Att2Str(layer_dict[sorted_edges[i].first]->at) << ") ";
        std::cerr << "-> " << sorted_edges[i].second; 
        
        std::cerr << " (" << Att2Str(layer_dict[sorted_edges[i].second]->at) << ") " << std::endl;
    }
		
	initialized = true;
	
	if (visited)
		delete[] visited;
    if (has_grad)
        delete[] has_grad;
	visited = new bool[layer_dict.size()];
    has_grad = new bool[layer_dict.size()];
	name_idx_map.clear();
	int cur_idx = 0;
	for (layeriter it = layer_dict.begin(); it != layer_dict.end(); ++it)
	{
		name_idx_map[it->first] = cur_idx;
		cur_idx++;
	}
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::ForwardData(std::map<std::string, GraphData<mode, Dtype>* > graph_input, Phase phase)
{
	if (!initialized)
		InitializeGraph(); 
	memset(visited, false, sizeof(bool) * layer_dict.size());	
	memset(has_grad, false, sizeof(bool) * layer_dict.size());
	for (paramiter it = param_dict.begin(); it != param_dict.end(); ++it)
		it->second->batch_prepared = false;
		
	for (dataiter it = graph_input.begin(); it != graph_input.end(); ++it)
	{
		layer_dict[it->first]->graph_output = it->second;
	}
	
	int k;
	for (size_t i = 0; i < sorted_edges.size(); ++i)
	{
		k = name_idx_map[sorted_edges[i].second];
		layer_dict[sorted_edges[i].second]->UpdateOutput(layer_dict[sorted_edges[i].first], visited[k] ? SvType::ADD2 : SvType::WRITE2, phase);              
		visited[k] = true;
        if (false)
        {
            std::cerr << layer_dict[sorted_edges[i].first]->name << " " << layer_dict[sorted_edges[i].second]->name << " ";
            auto* states = layer_dict[sorted_edges[i].second]->graph_output;            
            if (states->node_states)
                std::cerr << states->node_states->DenseDerived().Asum();
            else
                std::cerr << "null";
            std::cerr << " ";
            if (states->edge_states)
                std::cerr << states->edge_states->DenseDerived().Asum();
            else
                std::cerr << "null";              
            std::cerr << std::endl;                                      
        } 
	}
}

template<MatMode mode, typename Dtype>
std::map<std::string, Dtype> GraphNN<mode, Dtype>::ForwardLabel(std::map<std::string, GraphData<mode, Dtype>* > graph_truth)
{
	std::map<std::string, Dtype> loss;	
	for (dataiter it = graph_truth.begin(); it != graph_truth.end(); ++it)
	{
		auto* criterion_layer = dynamic_cast<ICriterionLayer<mode, Dtype>*>(layer_dict[it->first]);
		if (criterion_layer)
		{
			loss[it->first] = criterion_layer->GetLoss(it->second); 
            has_grad[name_idx_map[it->first]] = true;
		} else {
			throw "not a valid criterion_layer feedforward";
		}
	}
	
	return loss;
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::BackPropagation()
{
	memset(visited, false, sizeof(bool) * layer_dict.size());
	int k;
	for (std::vector< std::pair<std::string, std::string> >::reverse_iterator it = sorted_edges.rbegin(); it != sorted_edges.rend(); ++it)
	{
		k = name_idx_map[it->first];
		if (layer_dict[it->second]->properr != PropErr::T)
			continue;
        if (!has_grad[name_idx_map[it->second]])
            continue;
		if (layer_dict[it->first]->properr == PropErr::T)
			layer_dict[it->second]->BackPropErr(layer_dict[it->first], visited[k] ? SvType::ADD2 : SvType::WRITE2);
		visited[k] = true;
		has_grad[k] = true;
        layer_dict[it->second]->AccDeriv(layer_dict[it->first]);
	}
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum)
{
	for (paramiter it = param_dict.begin(); it != param_dict.end(); ++it)
		it->second->UpdateParams(lr, l2_penalty, momentum);
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::Save(std::string filename)
{
	FILE* fid = fopen(filename.c_str(), "wb");
	this->Serialize(fid);
	fclose(fid);
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::Load(std::string filename)
{
	FILE* fid = fopen(filename.c_str(), "rb");
	this->Deserialize(fid);
	fclose(fid);
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::Serialize(FILE* fid)
{
	for (paramiter it = param_dict.begin(); it != param_dict.end(); ++it)
    {
		it->second->Serialize(fid);
    }
}

template<MatMode mode, typename Dtype>
void GraphNN<mode, Dtype>::Deserialize(FILE* fid)
{
	for (paramiter it = param_dict.begin(); it != param_dict.end(); ++it)
    {
		it->second->Deserialize(fid);
    }		
}

template class GraphNN<CPU, float>;
template class GraphNN<CPU, double>;
template class GraphNN<GPU, float>;
template class GraphNN<GPU, double>;
