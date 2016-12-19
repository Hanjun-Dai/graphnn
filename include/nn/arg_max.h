#ifndef ARG_MAX_H
#define ARG_MAX_H

#include "util/gnn_macros.h"
#include "fmt/printf.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

template<typename mode, typename Dtype>
class ArgMax : public Factor
{
public:
	static std::string StrType()
	{
		return "ArgMax";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, int> >;
	
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, int> >(out_name);
	}

	ArgMax(std::string _name, PropErr _properr = PropErr::T);
};

}


#endif