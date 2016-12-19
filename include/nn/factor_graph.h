#ifndef FACTOR_GRAPH_H
#define FACTOR_GRAPH_H

#include "util/gnn_macros.h"
#include "nn/variable.h"
#include "fmt/printf.h"
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

	void AddVar(std::shared_ptr<Variable> var);

	std::vector<std::shared_ptr<Variable> > FeedForward(std::initializer_list<std::shared_ptr< Variable> > targets, 
														std::map<std::string, void*> feed_dict,
														uint n_thread = 1);

	void BackPropagate(std::initializer_list<std::shared_ptr< Variable> > targets, uint n_thread = 1);
	std::map<std::string, std::shared_ptr<Variable> > vars;
};

template<typename VarType, typename... Args>
std::shared_ptr<VarType> add_var(FactorGraph& g, std::string var_name, Args&&... args)
{
	auto v = std::make_shared<VarType>(var_name, std::forward<Args>(args)...);
	g.AddVar(v);
	return v;
}

template<typename FacType, typename VarPtr, typename... Args>
typename FacType::OutType af(FactorGraph& g, std::vector< VarPtr > operands, 
									Args&&... args)
{
	auto fname = fmt::sprintf("%s_%d", FacType::StrType(), g.vars.size());
	auto f = std::make_shared<FacType>(fname, std::forward<Args>(args)...);
	//g.AddFactor(f);
	return f->CreateOutVar();
}

template<typename FacType, typename VarPtr, typename... Args>
typename FacType::OutType af(std::initializer_list< VarPtr > op, Args&&... args)
{
	auto first_op = *(op.begin());
	auto* g = first_op->g;
	auto fname = fmt::sprintf("%s_%d", FacType::StrType(), g->vars.size());
	auto f = std::make_shared<FacType>(fname, std::forward<Args>(args)...);
	//g.AddFactor(f);
	return f->CreateOutVar();
}

}

#endif