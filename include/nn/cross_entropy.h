#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "util/gnn_macros.h"
#include "fmt/printf.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

template<typename mode, typename Dtype>
class CrossEntropy : public Factor
{
public:
	static std::string StrType()
	{
		return "CrossEntropy";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	CrossEntropy(std::string _name, bool _need_softmax, PropErr _properr = PropErr::T);

	bool need_softmax;
};

}

#endif