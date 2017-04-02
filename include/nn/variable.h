#ifndef VARIABLE_H
#define VARIABLE_H

#include <string>
#include <vector>

#include "fmt/format.h"
#include "tensor/tensor.h"
#include "tensor/cpu_dense_tensor.h"
#include "tensor/gpu_dense_tensor.h"
#include "tensor/cpu_sparse_tensor.h"
#include "tensor/gpu_sparse_tensor.h"

namespace gnn
{

class FactorGraph;

/**
 * @brief      interface for differentiable variables
 */
class IDifferentiable
{
public:
	/**
	 * @brief      clear gradient
	 */
	virtual void ZeroGrad() = 0;

	/**
	 * @brief      set the gradient to be ones; used on the top variable
	 */
	virtual void OnesGrad() = 0;
};

/**
 * @brief      the abstract class of variable; Variables are the objects which hold
 * 				the inputs to the operators, as well as the outputs from operators
 */
class Variable
{
public:
	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name  The variable string name
	 */
	Variable(std::string _name);

	/**
	 * @brief      use reference to set the variable; by doing so, we can avoid extra 
	 * 				copy work
	 *
	 * @param      p     a pointer which can be used to hold any object; however, a static_cast
	 * 					is required when use this
	 */
	virtual void SetRef(void* p) NOT_IMPLEMENTED

	/**
	 * @brief      Get the basic element type (float/double/int) of this variable
	 *             
	 * @return     The enum ele type.
	 */
	virtual EleType GetEleType() = 0;

	/**
	 * the variable's string name
	 */
	std::string name;
};

/**
 * @brief      Class for void pointer variable.
 */
class VoidPtrVar : public Variable
{
public:
	VoidPtrVar(std::string _name);

	virtual EleType GetEleType() override;

	virtual void SetRef(void* p) override;	

	/**
	 * the actual void ptr
	 */
	void* void_ptr;
};


class GraphStruct;

/**
 * @brief      Class for graph variable
 */
class GraphVar : public Variable
{
public:
	GraphVar(std::string _name);

	virtual EleType GetEleType() override;

	virtual void SetRef(void* p) override;	

	/**
	 * the actual graph
	 */
	GraphStruct* graph;
};

template<typename mode, typename matType, typename Dtype>
class TensorVarTemplate;

template<typename mode, typename Dtype>
using DTensorVar = TensorVarTemplate<mode, DENSE, Dtype>;

template<typename mode, typename Dtype>
using SpTensorVar = TensorVarTemplate<mode, SPARSE, Dtype>;

/**
 * @brief      Class for tensor variable, which is the most common variable in this package
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double/int }
 */
template<typename mode, typename Dtype>
class TensorVar : public Variable
{
public:
	/**
	 * @brief      Get the specific derived tensor variable type (DENSE/SPARSE)
	 *
	 * @tparam     matType  whether the derived tensor is DENSE/SPARSE
	 *
	 * @return     the derived subclass
	 */
	template<typename matType>
	TensorVarTemplate<mode, matType, Dtype>& Derived()
	{
		auto* ret = dynamic_cast<TensorVarTemplate<mode, matType, Dtype>*>(this);
		ASSERT(ret, "wrong derived type of TensorVar");
		return *ret;
	}

	virtual EleType GetEleType() override
	{
		return Dtype2Enum<Dtype>();
	}

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name  The name
	 */
	TensorVar(std::string _name) : Variable(_name) {}

	/**
	 * @brief      convert the tensor variable to a scalar; assert that the tensor is trivial here
	 *
	 * @return     the only element in this tensor
	 */
	virtual Dtype AsScalar() = 0;

	/**
	 * @brief      Gets the matrix type (DENSE/SPARSE)
	 *             
	 * @return     The matrix enum type
	 */
	virtual MatType GetMatType() = 0;
};

/**
 * @brief      implementation of TensorVar;
 *
 * @tparam     mode     { CPU/GPU }
 * @tparam     matType  { DENSE/SPARSE }
 * @tparam     Dtype    { float/double/int }
 */
template<typename mode, typename matType, typename Dtype>
class TensorVarTemplate : TensorVar<mode, Dtype> {};

/**
 * @brief      DENSE tensor specialization of TensorVar
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double/int }
 */
template<typename mode, typename Dtype>
class TensorVarTemplate<mode, DENSE, Dtype> : public TensorVar<mode, Dtype>, public IDifferentiable
{
public: 
	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name     The name
	 */
	TensorVarTemplate(std::string _name);

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name     The name
	 * @param[in]  l         the tensor shape specified by the size_t list
	 */
	TensorVarTemplate(std::string _name, std::vector<size_t> l);

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name  The name
	 * @param[in]  l      the tensor shape specified by int list
	 */
	TensorVarTemplate(std::string _name, std::vector<int> l)
		: TensorVarTemplate(_name, std::vector<size_t>(l.begin(), l.end())) {}

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name  The name
	 * @param[in]  l      the tensor shape specified by unsigned int list
	 */
	TensorVarTemplate(std::string _name, std::vector<uint> l)
		: TensorVarTemplate(_name, std::vector<size_t>(l.begin(), l.end())) {}

	/**
	 * @brief      Sets the reference.
	 *
	 * @param      p    Here we assume p is a raw pointer to DTensor<mode, Dtype>
	 */
	virtual void SetRef(void* p) override;

	virtual Dtype AsScalar() override;
	virtual MatType GetMatType() override;

	/**
	 * @brief     init the gradient to be zero
	 */
	virtual void ZeroGrad() override;

	virtual void OnesGrad() override;

	/**
	 * @brief      save to disk
	 *
	 * @param      fid   The file handle
	 */
	void Serialize(FILE* fid);

	/**
	 * @brief      load from disk
	 *
	 * @param      fid   The file handle
	 */
	void Deserialize(FILE* fid);
	
	/**
	 * the actual value of this variable
	 */
	DTensor< mode, Dtype > value;
	/**
	 * stores the gradient with respect to this variable
	 */
	DTensor< mode, Dtype> grad;
};

/**
 * @brief      SPARSE tensor specialization of TensorVar
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double/int }
 */
template<typename mode, typename Dtype>
class TensorVarTemplate<mode, SPARSE, Dtype> : public TensorVar<mode, Dtype>
{
public:
	TensorVarTemplate(std::string _name);
	/**
	 * @brief      Sets the reference.
	 *
	 * @param      p     Here we assume p is a raw pointer to SpTensor<mode, Dtype>
	 */
	virtual void SetRef(void* p) override;
	virtual Dtype AsScalar() override;
	virtual MatType GetMatType() override;

	/**
	 * the actual value of this variable
	 */
	SpTensor<mode, Dtype> value;
};

}
#endif
