#ifndef FACTOR_GRAPH_H
#define FACTOR_GRAPH_H

#include "util/gnn_macros.h"
#include "nn/variable.h"
#include <initializer_list>
#include <string>
#include <map>

namespace gnn
{

class Variable;

class FactorGraph
{
public:
	FactorGraph();	

	void Run(std::initializer_list<std::string> targets, uint n_thread = 1);
	void Run(std::initializer_list<Variable*> targets, uint n_thread = 1);
	void AddVar(std::shared_ptr<Variable> var);

	std::map<std::string, std::shared_ptr<Variable> > vars;
};

template<typename VarType, typename... Args>
std::shared_ptr<VarType> add_var(FactorGraph& g, std::string var_name, Args&&... args)
{
	auto v = std::make_shared<VarType>(var_name, std::forward<Args>(args)...);
	g.AddVar(v);
	return v;
}

// template<typename FacType, typename... Args>
// std::shared_ptr<FacType> add_factor(FactorGraph& g, Args&&... args)
// {
// 	auto fname = fmt::sprintf("%s_%d", FacType::StrType(), g.vars.size());
// 	auto f = std::make_shared<FacType>(fname, std::forward<Args>(args)...);
// 	g.AddFactor(f);
// }

}

#endif