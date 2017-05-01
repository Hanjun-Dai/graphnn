#ifndef L2_COL_NORM_H
#define L2_COL_NORM_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

template<typename Dtype>
void L2ColNormFwd(DTensor<CPU, Dtype>& in, DTensor<CPU, Dtype>& out, DTensor<CPU, Dtype>& norm2, DTensor<CPU, Dtype>& len);

template<typename Dtype>
void L2ColNormFwd(DTensor<GPU, Dtype>& in, DTensor<GPU, Dtype>& out, DTensor<GPU, Dtype>& norm2, DTensor<GPU, Dtype>& len);

template<typename Dtype>
void L2ColNormGrad(DTensor<CPU, Dtype>& x, DTensor<CPU, Dtype>& prev_grad, DTensor<CPU, Dtype>& cur_grad, DTensor<CPU, Dtype>& norm2, DTensor<CPU, Dtype>& len);

template<typename Dtype>
void L2ColNormGrad(DTensor<GPU, Dtype>& x, DTensor<GPU, Dtype>& prev_grad, DTensor<GPU, Dtype>& cur_grad, DTensor<GPU, Dtype>& norm2, DTensor<GPU, Dtype>& len);

/**
 * @brief      normalize each row of the matrix
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { ele_type (float/double) }
 */
template<typename mode, typename Dtype>
class L2ColNorm : public Factor
{
public:
	static std::string StrType()
	{
		return "L2ColNorm";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     a tensor with the same shape as inputs
	 */
	OutType CreateOutVar()
	{
		auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name     The name
	 * @param[in]  _properr  The properr
	 */
	L2ColNorm(std::string _name, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;

private:
	DTensor<mode, Dtype> norm2, len;
};

}

#endif