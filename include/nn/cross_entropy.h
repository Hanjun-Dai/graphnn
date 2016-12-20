#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "util/gnn_macros.h"
#include "fmt/printf.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

template<typename Dtype>
void CalcCrossEntropy(DTensor<CPU, Dtype>& prob, SpTensor<CPU, Dtype>& label, DTensor<CPU, Dtype>& out);

template<typename Dtype>
void CalcCrossEntropy(DTensor<GPU, Dtype>& prob, SpTensor<GPU, Dtype>& label, DTensor<CPU, Dtype>& out);

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
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs) override;

	bool need_softmax;
	DTensor<mode, Dtype> probs;
};

}

#endif