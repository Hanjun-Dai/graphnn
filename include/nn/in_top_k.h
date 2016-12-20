#ifndef IN_TOP_H_H
#define IN_TOP_H_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"
#include "fmt/printf.h"

namespace gnn
{

template<typename Dtype>
void IsInTopK(DTensor<CPU, Dtype>& pred, DTensor<CPU, int>& label, DTensor<CPU, int>& out, int k);

template<typename Dtype>
void IsInTopK(DTensor<GPU, Dtype>& pred, DTensor<GPU, int>& label, DTensor<CPU, int>& out, int k);

template<typename mode, typename Dtype>
class InTopK : public Factor
{
public:
	static std::string StrType()
	{
		return "InTopK";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, int> >;
	
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, int> >(out_name);
	}

	InTopK(std::string _name, int _topK = 1);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs) override;

	int topK;
};

}

#endif