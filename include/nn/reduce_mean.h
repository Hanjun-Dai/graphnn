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

	ReduceMean(std::string _name, PropErr _properr = PropErr::T);
};

}

#endif