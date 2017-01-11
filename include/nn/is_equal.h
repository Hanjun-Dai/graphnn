#ifndef IS_EQUAL_H
#define IS_EQUAL_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

template<typename mode, typename Dtype>
class IsEqual : public Factor
{
public:
	static std::string StrType()
	{
		return "IsEqual";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	IsEqual(std::string _name);

};

}


#endif