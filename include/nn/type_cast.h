#ifndef TYPE_CAST_H
#define TYPE_CAST_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"
#include "fmt/printf.h"

namespace gnn
{

template<typename mode, typename Dtype>
class TypeCast : public Factor
{
public:
	static std::string StrType()
	{
		return "Cast";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	TypeCast(std::string _name, PropErr _properr = PropErr::T);
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs) override;

};

}


#endif