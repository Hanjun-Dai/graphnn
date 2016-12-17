#include "nn/factor_graph.h"
#include "nn/variable.h"

namespace gnn
{

FactorGraph::FactorGraph()
{
	vars.clear();
}

void FactorGraph::Run(std::initializer_list<std::string> targets, uint n_thread)
{

}

void FactorGraph::Run(std::initializer_list<Variable*> targets, uint n_thread)
{

}

void FactorGraph::AddVar(std::shared_ptr<Variable> var)
{

}

}