#include "nn/msg_pass.h"

namespace gnn
{

template<typename mode, typename Dtype>
IMsgPass<mode, Dtype>::IMsgPass(std::string _name) : Factor(_name, PropErr::N), cpu_weight(nullptr)
{

}

template<typename mode, typename Dtype>
void IMsgPass<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< std::shared_ptr<Variable> >& outputs) 
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

}

template class IMsgPass<CPU, double>;
template class IMsgPass<CPU, float>;

}