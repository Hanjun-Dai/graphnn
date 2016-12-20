#ifndef RELU_H
#define RELU_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include <memory>
#include "nn/variable.h"

namespace gnn
{

template<typename Dtype>
void ReLUAct(DTensor<CPU, Dtype>& in, DTensor<CPU, Dtype>& out);

template<typename Dtype>
void ReLUAct(DTensor<GPU, Dtype>& in, DTensor<GPU, Dtype>& out);

template<typename mode, typename Dtype>
class ReLU : public Factor
{
public:
	static std::string StrType()
	{
		return "ReLU";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	OutType CreateOutVar()
	{
		auto out_name = this->name + ":out_0";
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	ReLU(std::string _name, PropErr _properr = PropErr::T);
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs) override;
	
};

}

#endif