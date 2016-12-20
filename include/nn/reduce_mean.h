#ifndef REDUCE_MEAN_H
#define REDUCE_MEAN_H

#include "util/gnn_macros.h"
#include "fmt/printf.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

template<typename mode, typename Dtype>
class ReduceMean : public Factor
{
public:
	static std::string StrType()
	{
		return "ReduceMean";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	ReduceMean(std::string _name, int _axis = -1, bool _keep_dim = false, PropErr _properr = PropErr::T);
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs) override;

	int axis;
	bool keep_dim;
};

}

#endif