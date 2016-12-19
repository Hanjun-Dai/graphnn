#include "nn/factor_graph.h"
#include "nn/variable.h"

namespace gnn
{

FactorGraph::FactorGraph()
{
	vars.clear();
}

std::vector<std::shared_ptr<Variable> > FactorGraph::FeedForward(std::initializer_list<std::shared_ptr<Variable> > targets, 
																std::map<std::string, void*> feed_dict,
																uint n_thread)
{
	std::vector<std::shared_ptr<Variable> > result(targets.size());

	return result;
}

void FactorGraph::BackPropagate(std::initializer_list<std::shared_ptr< Variable> > targets, 
								uint n_thread)
{
	
}								

void FactorGraph::AddVar(std::shared_ptr<Variable> var)
{

}

}