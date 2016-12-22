#ifndef MSG_PASS_H
#define MSG_PASS_H

#include "util/gnn_macros.h"
#include "util/graph_struct.h"
#include "fmt/format.h"
#include "nn/factor.h"
#include "nn/variable.h"
#include <memory>

namespace gnn
{

/**
 * @brief      construct a sparse matrix from graph; used for message passing; 
 *
 * @tparam     mode   CPU/GPU
 * @tparam     Dtype  float/double
 */
template<typename mode, typename Dtype>
class IMsgPass : public Factor
{
public:
	static std::string StrType()
	{
		return "IMsgPass";
	}
	
	IMsgPass(std::string _name);
	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;
protected:
	virtual void InitCPUWeight(GraphStruct* graph) = 0;
	SpTensor<CPU, Dtype>* cpu_weight;
};



}

#endif