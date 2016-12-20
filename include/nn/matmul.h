#ifndef MATMUL_H
#define MATMUL_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"
#include "fmt/printf.h"

namespace gnn
{

template<typename mode, typename Dtype>
class MatMul : public Factor
{
public:
	static std::string StrType()
	{
		return "MatMul";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	MatMul(std::string _name, Trans _transA = Trans::N, 
			Trans _transB = Trans::N, PropErr _properr = PropErr::T);
	
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs) override;

	Trans transA, transB;
};

}

#endif