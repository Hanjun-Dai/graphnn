#include "nn/matmul.h"

namespace gnn
{

template<typename mode, typename Dtype>
MatMul<mode, Dtype>::MatMul(std::string _name, PropErr _properr) 
		: Factor(_name, _properr)
{

}


}