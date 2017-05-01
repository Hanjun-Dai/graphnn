#ifndef MSG_PASS_H
#define MSG_PASS_H

#include "util/gnn_macros.h"
#include "util/graph_struct.h"
#include "nn/factor.h"
#include "nn/variable.h"
#include <memory>

namespace gnn
{

/**
 * @brief      construct a sparse matrix from graph; used for message passing; 
 *
 * @tparam     mode   CPU/GPU
 * @tparam     Dtype  float/double
 */
template<typename mode, typename Dtype>
class IMsgPass : public Factor
{
public:
	static std::string StrType()
	{
		return "IMsgPass";
	}
	
	IMsgPass(std::string _name, bool _average);
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< std::shared_ptr<Variable> >& outputs, 
						Phase phase) override;
protected:
	/**
	 * @brief      setup the sparse matrix on cpu
	 *
	 * @param      graph  The graph structure
	 */
	virtual void InitCPUWeight(GraphStruct* graph) = 0;

	/**
	 * a pointer to cpu_weight; when this factor runs on cpu, the pointer
	 * will point to the actual output, otherwise, we create a new matrix
	 * on cpu
	 */
	SpTensor<CPU, Dtype>* cpu_weight;
	bool average;
};

template<typename mode, typename Dtype>
class Node2NodeMsgPass : public IMsgPass<mode, Dtype>
{
public:
	static std::string StrType()
	{
		return "Node2NodeMsgPass";
	}

	using OutType = std::shared_ptr< SpTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     return a sparse tensor
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< SpTensorVar<mode, Dtype> >(out_name);
	}

	Node2NodeMsgPass(std::string _name, bool _average = false) : IMsgPass<mode, Dtype>(_name, _average) {} 
protected:
	virtual void InitCPUWeight(GraphStruct* graph) override;

};

template<typename mode, typename Dtype>
class Edge2NodeMsgPass : public IMsgPass<mode, Dtype>
{
public:
	static std::string StrType()
	{
		return "Edge2NodeMsgPass";
	}

	using OutType = std::shared_ptr< SpTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     return a sparse tensor
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< SpTensorVar<mode, Dtype> >(out_name);
	}

	Edge2NodeMsgPass(std::string _name, bool _average = false) : IMsgPass<mode, Dtype>(_name, _average) {} 
protected:
	virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<typename mode, typename Dtype>
class Node2EdgeMsgPass : public IMsgPass<mode, Dtype>
{
public:
	static std::string StrType()
	{
		return "Node2EdgeMsgPass";
	}

	using OutType = std::shared_ptr< SpTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     return a sparse tensor
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< SpTensorVar<mode, Dtype> >(out_name);
	}

	Node2EdgeMsgPass(std::string _name, bool _average = false) : IMsgPass<mode, Dtype>(_name, _average) {} 
protected:
	virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<typename mode, typename Dtype>
class Edge2EdgeMsgPass : public IMsgPass<mode, Dtype>
{
public:
	static std::string StrType()
	{
		return "Edge2EdgeMsgPass";
	}

	using OutType = std::shared_ptr< SpTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     return a sparse tensor
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< SpTensorVar<mode, Dtype> >(out_name);
	}

	Edge2EdgeMsgPass(std::string _name, bool _average = false) : IMsgPass<mode, Dtype>(_name, _average) {} 
protected:
	virtual void InitCPUWeight(GraphStruct* graph) override;
};

template<typename mode, typename Dtype>
class SubgraphMsgPass : public IMsgPass<mode, Dtype>
{
public:
	static std::string StrType()
	{
		return "SubgraphMsgPass";
	}

	using OutType = std::shared_ptr< SpTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     return a sparse tensor
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< SpTensorVar<mode, Dtype> >(out_name);
	}

	SubgraphMsgPass(std::string _name, bool _average = false) : IMsgPass<mode, Dtype>(_name, _average) {} 
protected:
	virtual void InitCPUWeight(GraphStruct* graph) override;
};

}

#endif