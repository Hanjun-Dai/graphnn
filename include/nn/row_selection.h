#ifndef ROW_SELECTION_H
#define ROW_SELECTION_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

template<typename Dtype>
void RowSelectionFwd(DTensor<CPU, Dtype>& input, DTensor<CPU, Dtype>& output, DTensor<CPU, int>& row_idxes);

template<typename Dtype>
void RowSelectionBackwd(RowSpTensor<CPU, Dtype>& prev_grad, DTensor<CPU, Dtype>& cur_grad, DTensor<CPU, int>& row_idxes);

/**
 * @brief      Operator: row selection
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { ele_type (float/double) }
 */
template<typename mode, typename Dtype>
class RowSelection : public Factor
{
public:
	static std::string StrType()
	{
		return "RowSelection";
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
	RowSelection(std::string _name, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;
};

}
#endif