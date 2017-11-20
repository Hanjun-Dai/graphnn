#ifndef FACTOR_GRAPH_H
#define FACTOR_GRAPH_H

#include "util/gnn_macros.h"
#include "nn/variable.h"
#include "nn/factor.h"
#include <initializer_list>
#include <string>
#include <map>
#include <queue>

#include <tuple>
#include <array>

namespace gnn
{

class Variable;

}

template<int... Indices>
struct indices {
    using next = indices<Indices..., sizeof...(Indices)>;
};

template<int Size>
struct build_indices {
    using type = typename build_indices<Size - 1>::type::next;
};

template<>
struct build_indices<0> {
    using type = indices<>;
};

template<typename T>
using Bare = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template<typename Tuple>
constexpr
typename build_indices<std::tuple_size<Bare<Tuple>>::value>::type
make_indices()
{ return {}; }

template<typename Tuple, int... Indices>
std::array<
  std::shared_ptr<gnn::Variable>,
    std::tuple_size<Bare<Tuple>>::value
>
to_var_array(Tuple&& tuple, indices<Indices...>)
{
    using std::get;
    return {{ get<Indices>(std::forward<Tuple>(tuple))... }};
}

template<typename Tuple>
auto to_var_array(Tuple&& tuple)
-> decltype( to_var_array(std::declval<Tuple>(), make_indices<Tuple>()) )
{
    return to_var_array(std::forward<Tuple>(tuple), make_indices<Tuple>());
}

template<typename T>
std::array< std::shared_ptr<T>, 1> to_var_array(std::shared_ptr<T>& tuple)
{
	return {{ tuple }};
}

namespace gnn
{


/**
 * @brief      the computation graph; responsible for representing the factor graph, as well as the execution
 */
class FactorGraph
{
public:	
	typedef std::shared_ptr<Variable> VarPtr;
	typedef std::vector<std::shared_ptr<Variable> > VarList;
	typedef std::vector<std::shared_ptr<Factor> > FactorList;

	/**
	 * @brief      constructor
	 */
	FactorGraph();	

	/**
	 * @brief      Adds a constant variable to this computation graph
	 *
	 * @param[in]  var            The variable
	 * @param[in]  isPlaceholder  Indicates if placeholder
	 */
	void AddConst(VarPtr var, bool isPlaceholder);

	/**
	 * @brief      Add a parameter to the computation graph
	 *
	 * @param[in]  var   The parameter variable
	 */
	void AddParam(VarPtr var);

	/**
	 * @brief      Adds a factor (operator) to the graph
	 *
	 * @param[in]  factor      The factor
	 * @param[in]  vars_in     The variables (inputs) to the factor
	 * @param[in]  vars_out    The variables (outputs) produced by the factor
	 *
	 * @tparam     FactorPtr   { the factor class type }
	 * @tparam     VarPtrList  { for the variable class }
	 * @tparam     VarOutList  { for the variable class }
	 */
	template<typename FactorPtr, typename VarPtrList, typename VarOutList>
	void AddFactor(FactorPtr factor, VarPtrList vars_in, VarOutList vars_out)
	{
		// must be a new factor
		ASSERT(factor_dict.count(factor->name) == 0 && factorEdges.count(factor->name) == 0, 
				fmt::sprintf("factor %s is already inserted", factor->name.c_str())); 

		// check input arguments
		VarList in_list(vars_in.size());
		for (size_t i = 0; i < vars_in.size(); ++i)
		{
			ASSERT(var_dict.count(vars_in[i]->name), "the operand " << vars_in[i]->name << " must be registered beforehand");
			in_list[i] = vars_in[i];
		}

		// check output vars
		VarList out_list(vars_out.size());
		for (size_t i = 0; i < vars_out.size(); ++i)
		{
			AddVar(vars_out[i]);
			if (factor->properr == PropErr::N)
				isConst[VarIdx(vars_out[i])] = true;
			out_list[i] = vars_out[i];
		}
		
		// f -> out_var
		for (auto p : vars_out)
			varEdges[p->name].first.push_back(factor);
		// in_var -> f
		for (auto p : vars_in)
			varEdges[p->name].second.push_back(factor);

		// add factor
		factorEdges[factor->name] = std::make_pair(in_list, out_list);
		factor_dict[factor->name] = std::make_pair(factor_list.size(), factor);
		factor_list.push_back(factor);
	}

	/**
	 * @brief      { function_description }
	 *
	 * @param[in]  targets    The targets which the user wants to fetch
	 * @param[in]  feed_dict  The feed dictionary; used to set the placeholders	 
	 * @param[in]  phase      train/test
	 * @param[in]  n_thread   # threads used in this function
	 *
	 * @return     { The targets required by user }
	 */
	VarList FeedForward(std::vector< VarPtr > targets, 
						std::map<std::string, void*> feed_dict,
						Phase phase, 
						uint n_thread = 1);

	/**
	 * @brief      back propagation funciton; used to calculate the gradient with respect to each variable
	 *
	 * @param[in]  targets   The top variables that contributes to the objective 
	 * 						(typically only loss should be provided here)
	 * @param[in]  n_thread  # threads used in this function
	 */
	void BackPropagate(std::vector< VarPtr > targets, uint n_thread = 1);

	/**
	 * @brief      Variable index in this graph
	 *
	 * @param[in]  var   The shared_ptr to variable
	 *
	 * @return     { the integer index }
	 */
	size_t VarIdx(VarPtr var);

	/**
	 * @brief      Variable index in this graph
	 *
	 * @param[in]  var_name  The variable name
	 *
	 * @return     { the integer index }
	 */
	size_t VarIdx(std::string var_name);

	/**
	 * @brief      Factor index in this graph
	 *
	 * @param[in]  fac   The shared_ptr to factor
	 *
	 * @return     { the integer index }
	 */
	size_t FacIdx(std::shared_ptr<Factor> fac);	
	
	/**
	 * the map: string name -> (variable index, variable ptr)
	 */
	std::map<std::string, std::pair< size_t, VarPtr> > var_dict;

	/**
	 * the map: string name -> (factor index, factor ptr)
	 */
	std::map<std::string, std::pair< size_t, std::shared_ptr<Factor> > > factor_dict;

	/**
	 * variable list; used for integer indexing
	 */
	std::vector< VarPtr > var_list;

	/**
	 * factor list; used for integer indexing
	 */
	std::vector< std::shared_ptr< Factor > > factor_list;	
	/**
	 * the in/out edges of a factor: factor_name -> (in variables, out variables)
	 */
	std::map<std::string, std::pair< VarList, VarList > > factorEdges;
	/**
	 * the in/out edges of an variable: variable_name -> (factor(s) produce this var, factors use this var)
	 */
	std::map<std::string, std::pair< FactorList, FactorList > > varEdges;
	
protected:
	/**
	 * @brief      Adds an intermediate variable to this computation graph
	 *
	 * @param[in]  var        The variable to be add
	 */
	void AddVar(VarPtr var);

	/**
	 * @brief      The single threaded feed forward function
	 *
	 * @param[in]  targets    The targets which the user wants to fetch
	 * @param[in]  feed_dict  The feed dictionary; used to set the placeholders
	 * @param[in]  phase	  train/test
	 */
	void SequentialForward(std::vector< VarPtr > targets, 
							std::map<std::string, void*> feed_dict, 
							Phase phase);

	/**
	 * @brief      The single threaded backward function
	 *
	 * @param[in]  targets  The targets which contributes directly to the objective
	 */
	void SequentialBackward(std::vector< VarPtr > targets);
	
	/**
	 * @brief      Parse the dependency to see which variables are required
	 *
	 * @param[in]  targets  The targets which the user wants to fetch
	 */
	void DependencyParse(std::vector< VarPtr > targets);

	/**
	 * whether the variable (indexed by name) is ready
	 */
	std::map<std::string, VarPtr> ready_dict;

	/**
	 * whether the variable (indexed by integer) is ready
	 */
	std::vector<bool> isReady;

	/**
	 * whether the variable is constant
	 */
	std::vector<bool> isConst;

	/**
	 * whether the variable (indexed by integer) is required by user
	 */
	std::vector<bool> isRequired;

	/**
	 * num of pending variables each factor has; when it is zero, this factor can be executed
	 */
	std::vector<size_t> n_pending;

	/**
	 * whether the factor is executed during feedforward
	 */
	std::vector<size_t> isFactorExecuted;

	/**
	 * num of pending factors each var has during bp;
	 */
	std::vector<size_t> var_bp_pendings;

	/**
	 * whether the variable has gradient
	 */
	std::vector<size_t> var_has_grad;	

	/**
	 * queue data structure used for topo_sort/BFS
	 */
	std::queue<std::string> q;

	template<typename FacType, typename OprList, typename... Args>
	friend typename FacType::OutType af(FactorGraph& g, OprList operands, 
										Args&&... args)
	{
		auto fname = fmt::sprintf("%s_%d", FacType::StrType(), g.factorEdges.size());
		auto f = std::make_shared<FacType>(fname, std::forward<Args>(args)...);
		auto out_vars = f->CreateOutVar();

		auto vars_out = to_var_array( out_vars );
		g.AddFactor(f, operands, vars_out);	
		
		return out_vars;
	}
};

template<typename VarType, typename... Args>
std::shared_ptr<VarType> add_const(FactorGraph& g, std::string var_name, bool isPlaceholder, Args&&... args)
{
	auto v = std::make_shared<VarType>(var_name, std::forward<Args>(args)...);
	g.AddConst(v, isPlaceholder);
	return v;
}

template<typename FacType, typename VarPtr, typename... Args>
typename FacType::OutType af(FactorGraph& g, std::initializer_list< VarPtr > op, Args&&... args)
{
	std::vector< VarPtr > op_list;
	for (auto p : op)
		op_list.push_back(p);

	return af<FacType>(g, op_list, std::forward<Args>(args)...);
}

template<typename FacType, typename VarPtr1, typename VarPtr2, typename... Args>
typename FacType::OutType af(FactorGraph& g, std::pair<VarPtr1, VarPtr2> op, Args&&... args)
{
	std::vector< std::shared_ptr<Variable> > op_list(2);
	op_list[0] = op.first;
	op_list[1] = op.second;

	return af<FacType>(g, op_list, std::forward<Args>(args)...);
}

}

#endif